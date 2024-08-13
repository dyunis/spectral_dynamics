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
            # TODO if has GradientProjector callback, add back the proj_off component before logging svd
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
                        if getattr(mod, k, None):
                            _name = f'{name}.{k}'
                            weight = getattr(mod, k)
                            s = torch.linalg.svdvals(weight)
                            for i in range(len(s)):
                                self.svs[name][i].append(float(s[i]))
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

        # save self.svs as numpy array
        np_svs = {}
        for k in self.svs.keys():
            svs = []
            for ix in self.svs[k]:
                svs.append(self.svs[k][ix])
            svs = np.stack(svs)
            np_svs[k] = svs
        np.savez(f'{savedir}/svs.npz', **np_svs)


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
        steps_per_epoch = len(trainer.train_dataloader)
        num_ckpts = 5
        first_epoch_save_interval = len(trainer.train_dataloader) // num_ckpts
        if trainer.global_step < len(trainer.train_dataloader) and trainer.global_step // first_epoch_save_interval > 0 and trainer.global_step // first_epoch_save_interval < num_ckpts and trainer.global_step % first_epoch_save_interval == 0:
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
        assert perturb_type in ('random')
        self.perturb_type = perturb_type
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.perturb_epoch:
            self.perturb_(pl_module)
            # fix the batchnorm now that parameters are perturbed
            if has_bn(pl_module):
                device = next(pl_module.parameters()).device
                torch.optim.swa_utils.update_bn(trainer.train_dataloader, pl_module.model, device)

    @torch.no_grad()
    def perturb_(self, model):
        device = next(model.parameters()).device
        named_mat_params = [(n, p) for n, p in model.named_parameters() if len(p.shape) >= 2]
        # skip perturbing last linear layer as it will have outsized impact on feature dynamics
        named_mat_params = named_mat_params[:-1]
        for n, p in named_mat_params:
            mat = p
            if len(p.shape) > 2:
                mat = p.flatten(1)

            # on cpu for memory reasons (VGG)
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


# IMP masking from some specified final checkpoint path, then train with mask
# by projecting params and gradients to mask after every gradient step
class ModelMasker(Callback):
    def __init__(self, ckpt_dir, mask_epoch, mask_sparsity=0.1, mask_type='magnitude', seed=0):
        assert os.path.exists(ckpt_dir)
        cp_paths = glob(os.path.join(ckpt_dir, '*.ckpt'))
        cp_paths = [p for p in cp_paths if os.path.basename(p) != 'last.ckpt']
        # sort pytorch lightning ckpts
        cp_paths = sorted(cp_paths, key=pl_ckpt_path_to_step)
        mask_ckpt_path = cp_paths[-1]

        self.mask_epoch = mask_epoch
        self.mask_sparsity = mask_sparsity
        self.seed = seed
        assert mask_type in ('magnitude', 'reinit', 'random')
        self.mask_type = mask_type
        self.masks = self.get_masks(mask_ckpt_path)
        self.use_masks = False

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.mask_epoch:
            self.use_masks = True
            # reinit once to throw off masking pattern
            if self.mask_type == 'reinit':
                self.reinit_(pl_module)
            self.mask_(pl_module)
            # fix the batchnorm now that parameters are masked
            if has_bn(pl_module):
                device = next(pl_module.parameters()).device
                torch.optim.swa_utils.update_bn(trainer.train_dataloader, pl_module.model, device)

    # first batch will not have masking, but this way the saved checkpoints will
    # off-by-one wrt original lth convention:
    # https://github.com/facebookresearch/open_lth/issues/6
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.use_masks:
            self.mask_(pl_module)

    @torch.no_grad()
    def mask_(self, pl_module):
        for n, p in pl_module.named_parameters():
            for k, mask in self.masks.items():
                if n == k:
                    # mask params
                    p.copy_(p * mask)
                    # mask grads
                    if p.grad is not None:
                        p.grad.copy_(p.grad * mask)

    @torch.no_grad()
    def reinit_(self, pl_module):
        param_keys = [n for n, p in pl_module.named_parameters() if len(p.shape) >= 2]
        param_keys = param_keys[:-1]  # don't mask final layer
        for n, p in pl_module.named_parameters():
            if n in param_keys:
                # see reset_parameters() in torch.nn.modules.linear.Linear
                # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
                # and reset_parameters() in torch.nn.modules.conv._ConvNd
                # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    # for non final layer, non biases
    def get_masks(self, mask_ckpt_path):
        sd = torch.load(mask_ckpt_path)['state_dict']
        param_keys = [n for n, p in sd.items() if len(p.shape) >= 2]
        param_keys = param_keys[:-1]  # don't mask final layer

        # get global threshold of magnitude (not layer-wise)
        magnitudes = torch.cat([sd[k].flatten() for k in param_keys]).flatten().abs()
        magnitudes = magnitudes.sort()[0]
        threshold = magnitudes[-int(self.mask_sparsity * len(magnitudes))]

        generator = torch.Generator(device=magnitudes.device).manual_seed(self.seed)
        masks = {}
        for k in param_keys:
            p = sd[k]
            magnitude_mask = (p.abs() > threshold).float()
            if self.mask_type == 'magnitude' or self.mask_type == 'reinit':
                masks[k] = magnitude_mask
            elif self.mask_type == 'random':
                # per layer random mask should have same sparsity as magnitude mask
                magnitude_sparsity = magnitude_mask.sum() / magnitude_mask.numel()
                masks[k] = (torch.rand(*p.shape, generator=generator, device=p.device) < magnitude_sparsity).float()
            else:
                raise NotImplementedError(f'mask type {self.mask_type} must be "magnitude" "random" or "reinit"')
        return masks


def pl_ckpt_path_to_step(ckpt_path):
    root, _ = os.path.splitext(os.path.basename(ckpt_path))
    step = int(root.split('-')[1].split('=')[1])
    return step


def has_bn(module):
    for name, mod in module.named_modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            return True
        elif isinstance(mod, torch.nn.BatchNorm2d):
            return True
        elif isinstance(mod, torch.nn.BatchNorm3d):
            return True
        elif isinstance(mod, torch.nn.GroupNorm):
            return True
    return False
