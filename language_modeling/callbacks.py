import argparse
from collections import defaultdict
from datetime import timedelta
from glob import glob
import math
import os
import time
from typing import Dict, Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, Timer
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import wandb


class PlWandbEvaluator(Callback):
    def __init__(self, model, dataloader, config, split='train'):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self._eval_freq = config.eval_interval
        self.split = split

    @torch.no_grad()
    def evaluate(self, trainer, pl_module):
        pl_module.eval()
        device = next(self.model.parameters()).device
        totals = defaultdict(lambda: 0)
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc='eval')):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            outputs = pl_module.validation_step(batch, batch_idx)
            for k in outputs.keys():
                totals[k] += float(outputs[k])
        for k in totals:
            totals[k] /= len(self.dataloader)
        pl_module.train()
        return totals

    @torch.no_grad()
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self._eval_freq == 0:
            totals = self.evaluate(trainer, pl_module)
            totals = {f'{self.split}/{k}': v for k, v in totals.items()}
            wandb.log(totals, step=trainer.current_epoch)
            print(totals)
        return True

    @torch.no_grad()
    def on_train_end(self, trainer, pl_module):
        totals = self.evaluate(trainer, pl_module)
        totals = {f'{self.split}/{k}': v for k, v in totals.items()}
        wandb.log(totals, step=trainer.current_epoch)
        print(totals)
        return True


class PlSVComputer(Callback):
    def __init__(self, config):
        self.config = config
        self._eval_freq = config.eval_interval
        self.svs = defaultdict(lambda: defaultdict(lambda: []))

    @torch.no_grad()
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self._eval_freq == 0:
            for name, mod in pl_module.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    weight = mod.weight
                    s = torch.linalg.svdvals(weight)
                    for i in range(len(s)):
                        self.svs[name][i].append(float(s[i]))
                elif isinstance(mod, torch.nn.Conv2d):
                    weight = mod.weight
                    # (c_out, c_in, h, w) is weight shape, last 3 dims
                    # are input dim
                    weight = weight.flatten(1)
                    s = torch.linalg.svdvals(weight)
                    for i in range(len(s)):
                        self.svs[name][i].append(float(s[i]))
                elif isinstance(mod, torch.nn.LSTM):
                    # params are stored inside LSTM module given the fused kernel
                    param_keys = [k for k in dir(mod) if k.startswith('weight_')]
                    for k in param_keys:
                        weight = getattr(mod, k)
                        param_name = f'{name}.{k}'
                        s = torch.linalg.svdvals(weight)
                        for i in range(len(s)):
                            self.svs[param_name][i].append(float(s[i]))
                elif isinstance(mod, torch.nn.MultiheadAttention):
                    # attn layers also have specific parameters possible
                    param_keys = ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
                    for k in param_keys:
                        if getattr(mod, k, None) is not None:
                            _name = f'{name}.{k}'
                            weight = getattr(mod, k)
                            s = torch.linalg.svdvals(weight)
                            for i in range(len(s)):
                                self.svs[_name][i].append(float(s[i]))
            self.plot_svs(trainer, pl_module)
        return True

    def on_train_end(self, trainer, pl_module):
        self.plot_svs(trainer, pl_module)
        return True

    def plot_svs(self, trainer, pl_module):
        for name in self.svs:
            if len(self.svs[name]) < 1:
                continue
            fig, ax = plt.subplots(tight_layout=True)
            num_svs = len(self.svs[name])

            colormap = matplotlib.colormaps['viridis'].resampled(num_svs)
            for i in range(num_svs):
                ax.plot(self.svs[name][i], color=colormap(i))
            savedir = trainer.checkpoint_callback.dirpath
            fig.savefig(f'{savedir}/{name}.png')
            plt.close(fig)


# do not reload state dict so that training can proceed in [duration] increments
# from https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/callbacks/timer.html#Timer
class PlSlurmTimer(Timer):
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        time_elapsed = {}
        self._offset = 0


# reseed dataloader every epoch for tight control on order
# combine with saving at the end of epoch for reproducibility
# tested against dataloader, __iter__ is called after train_start
class PlSeedCallback(Callback):
    def __init__(self, seed):
        self.seed = seed
        pl.seed_everything(self.seed)  # do this because __iter__ is called before on_train_epoch_start

    # this is called before __iter__ of dataloader so it's a good place
    # to reseed everything, need the +1 to differentiate between start
    def on_train_epoch_start(self, trainer, pl_module):
        pl.seed_everything(self.seed + 1 + trainer.current_epoch)


# overrides pl ModelCheckpoint to save initialization when training starts
# https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/model_checkpoint.html#ModelCheckpoint
class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        ckpt_paths = glob(os.path.join(self.dirpath, '*.ckpt'))
        if len(ckpt_paths) == 0:
            trainer.save_checkpoint(os.path.join(self.dirpath, 'epoch=0-step=0.ckpt'))

    # within first epoch save 5 checkpoints evenly spaced
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # unless we're saving all checkpoints, skip first epoch saving
        if self.save_top_k != -1:
            return
        steps_per_epoch = len(trainer.train_dataloader) // pl_module.config.accumulate_grad_batches
        num_ckpts = 5
        first_epoch_save_interval = len(trainer.train_dataloader) // num_ckpts // pl_module.config.accumulate_grad_batches
        if trainer.global_step < steps_per_epoch and trainer.global_step // first_epoch_save_interval > 0 and trainer.global_step // first_epoch_save_interval < num_ckpts and trainer.global_step % first_epoch_save_interval == 0:
            trainer.save_checkpoint(os.path.join(self.dirpath, f'epoch=0-step={trainer.global_step}.ckpt'))

    def on_train_epoch_end(self, trainer, pl_module):
        # skip saving midway through epoch so checkpoints are always aligned for multi-job runs
        slurm_timer = [c for c in trainer.callbacks if isinstance(c, PlSlurmTimer)][0]
        if slurm_timer.time_remaining() <= 0:
            return
        super().on_train_epoch_end(trainer, pl_module)


# perturb ranks of all but last weight matrix a single time in training
# with some alpha weight relative to original magnitude
# either top half or bottom half of model
class ModelPerturber(Callback):
    def __init__(self, perturb_epoch, scale=1.0, perturb_type='random', seed=0): 
        self.perturb_epoch = perturb_epoch
        self.scale = scale
        self.seed = seed
        self.perturb_bottom_half = perturb_bottom_half  # default to perturbing top ranks
        assert perturb_type in ('random')
        self.perturb_type = perturb_type
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.perturb_epoch:
            self.perturb_(pl_module)

    @torch.no_grad()
    def perturb_svd_neg_(self, model):
        device = next(model.parameters()).device
        named_mat_params = [(n, p) for n, p in model.named_parameters() if len(p.shape) >= 2]
        # skip perturbing last linear layer as it will have outsized impact on feature dynamics
        named_mat_params = named_mat_params[:-1]
        for n, p in named_mat_params:
            mat = p
            if len(p.shape) > 2:
                mat = p.flatten(1)

            if self.perturb_type == 'random':
                # random perturbation
                pert = torch.normal(0.0, 1.0, size=mat.size(), generator=self.generator, device='cpu')
            else:
                raise NotImplementedError(f'perturb type {self.perturb_type}')
            pert = pert / pert.norm()  # keep norms consistent for controls

            new_param = mat.cpu() + self.scale * mat.norm().cpu() * pert
            new_param = mat.norm().cpu() * new_param / new_param.norm()
            if len(p.shape) > 2:
                new_param = new_param.reshape(p.shape)
            new_param = new_param.to(p.device)  # cast back to gpu
            p.copy_(new_param)
