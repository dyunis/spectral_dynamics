from datetime import timedelta
from glob import glob
import json
import os
import random
import pickle
import time

import lightning.pytorch as pl
from lightning.pytorch.plugins.environments import LightningEnvironment, SLURMEnvironment
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import numpy as np
import torch

import callbacks
from dataset import get_dataset
from model import PlModel


def main(config, savedir):
    start_time = time.time()
    if rank_zero_only.rank == 0:
        save_config(config, savedir)

    # seed for deterministic setup
    pl.seed_everything(0)

    trainset, valset = get_dataset(config)
    # only evaluate 1000 examples of trainset
    rng = np.random.default_rng(12345)
    ixs = rng.permutation(len(trainset)).tolist()[:1000]
    eval_trainset = torch.utils.data.Subset(trainset, ixs)

    # construct dataset and pytorch dataloader for pl
    # num_workers=0 as long as data is sitting on gpu
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    eval_train_loader = torch.utils.data.DataLoader(eval_trainset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    pl.seed_everything(config.seed)

    # construct model
    model = PlModel(config, trainset.tokenizer)

    setup_duration = time.time() - start_time
    time_left = config.time_limit
    time_left -= setup_duration
    if time_left < 0:
        raise TimeoutError('Time limit {config.time_limit} has elapsed')
    seeder = callbacks.PlSeedCallback(config.seed)
    time_str = timedelta_to_str(timedelta(seconds=time_left))
    slurm_timer = callbacks.PlSlurmTimer(duration=time_str)
    callback = [seeder, slurm_timer]

    if rank_zero_only.rank == 0:
        saver = callbacks.ModelCheckpoint(
            dirpath=savedir,
            save_last=True,
            save_on_train_epoch_end=True,
            save_top_k=-1,
            every_n_epochs=config.save_interval,  # but only at certain interval
        )
        callback.append(saver)

    if config.use_wandb and rank_zero_only.rank == 0:
        from callbacks import PlWandbEvaluator
        train_evaluator = PlWandbEvaluator(model, eval_train_loader, config, split='train')
        val_evaluator = PlWandbEvaluator(model, val_loader, config, split='val')
        callback.extend([train_evaluator, val_evaluator])

    # plot svs over time
    if config.plot_svs and rank_zero_only.rank == 0:
        from callbacks import PlSVComputer
        sv_computer = PlSVComputer(config)
        callback.append(sv_computer)

    if config.perturb_epoch > -1:
        from callbacks import ModelPerturber
        perturber = ModelPerturber(config.perturb_epoch, scale=config.perturb_scale, perturb_type=config.perturb_type, seed=config.seed)
        callback.append(perturber)

    assert config.num_gpus > 0

    # we want to manually launch multigpu, not let sbatch handle it
    # see https://github.com/Lightning-AI/pytorch-lightning/issues/18650#issuecomment-1747669666
    plugins = None
    if SLURMEnvironment.detect():
        print('detected slurm env (likely launched via sbatch), using lightning env to allow multigpu')
        plugins = [LightningEnvironment()]

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator='gpu',
        num_nodes=1,
        devices=config.num_gpus,
        strategy='ddp' if config.num_gpus > 1 else 'auto',
        deterministic=True,
        callbacks=callback,
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        logger=None,
        plugins=plugins,
    )

    # if resuming, train for less time (while resetting optimizer)
    # one issue with this resuming is that it also resumes optimizer
    # state, does this affect LMC behavior?
    ckpt_path = 'last'
    if config.ckpt_path is not None:
        assert os.path.exists(config.ckpt_path)
        # symlink prior checkpoints before the epoch of config.ckpt_path
        # to make plotting easy for later scripts
        symlink_starting_trajectory_(config.ckpt_path, savedir)
        cp = torch.load(config.ckpt_path, map_location='cpu')
        completed_steps = cp['global_step']

        # start after cp
        last_ckpt_path = os.path.join(savedir, 'last.ckpt')
        if os.path.exists(last_ckpt_path):
            cp = torch.load(last_ckpt_path, map_location='cpu')
            last_completed_steps = cp['global_step']
            if last_completed_steps < completed_steps:
                ckpt_path = config.ckpt_path
            else:
                ckpt_path = 'last'
        else:
            ckpt_path = config.ckpt_path
    else:
        ckpt_path = 'last'

    trainer.fit(
        model,
        train_loader,
        ckpt_path=ckpt_path,
    )


def timedelta_to_str(td):
    days, hours, minutes = td.days, td.seconds // 3600, (td.seconds // 60) % 60
    assert days < 100
    seconds = td.seconds - hours * 3600 - minutes * 60
    day_str = f'{days:0>2}'
    hour_str = f'{hours:0>2}'
    min_str = f'{minutes:0>2}'
    sec_str = f'{seconds:0>2}'
    string = ':'.join([day_str, hour_str, min_str, sec_str])
    return string


def save_config(config, savedir):
    config_path = os.path.join(savedir, 'config.json')
    if not os.path.exists(config_path):
        if hasattr(config, 'as_dict'):
            config_dict = config.as_dict()
        else:
            config_dict = vars(config)
        config_dict['savedir'] = savedir  # this gets modified by wandb
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def symlink_starting_trajectory_(src_cp_path, dst):
    # find previous cps
    src_cp_path, dst = os.path.abspath(src_cp_path), os.path.abspath(dst)
    src = os.path.dirname(src_cp_path)
    cp_paths = glob(os.path.join(src, '*.ckpt'))
    cp_paths = [p for p in cp_paths if os.path.basename(p) != 'last.ckpt']
    # sort pytorch lightning ckpts
    cp_paths = sorted(cp_paths, key=pl_ckpt_path_to_step)
    last_ix = cp_paths.index(src_cp_path)
    cp_paths = cp_paths[:last_ix + 1]
    assert len(cp_paths) > 0
    # symlink previous cps to dst (for plotting)
    for p in cp_paths:
        if not os.path.exists(os.path.join(dst, os.path.basename(p))):
            os.symlink(p, os.path.join(dst, os.path.basename(p)))


def pl_ckpt_path_to_step(ckpt_path):
    root, _ = os.path.splitext(os.path.basename(ckpt_path))
    step = int(root.split('-')[1].split('=')[1])
    return step


if __name__=='__main__':
    from config import config, setup_wandb
    if config.use_wandb and rank_zero_only.rank == 0:
        import wandb
        config, wandb_dir, wandb_name, wandb_id, savedir = setup_wandb(config)
    else:
        savedir = config.savedir
        os.makedirs(savedir, exist_ok=True)
    try:
        main(config, savedir)
    except TimeoutError as e:
        print(str(e))  # otherwise clogs emails with slurm timeouts
    finally:
        if config.use_wandb and rank_zero_only.rank == 0:
            wandb.finish()
