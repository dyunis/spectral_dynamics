import argparse
from copy import copy
from collections import defaultdict
from glob import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from dataset import get_dataset
from model import PlModel


def main(config, run_configs):
    # run_configs should only differ by seed
    config_dicts = [vars(rc) for rc in run_configs]
    all_hyps = {k: list(set([cd[k] for cd in config_dicts])) for k in config_dicts[0]}
    swept_hyps = [k for k in all_hyps if len(all_hyps[k]) > 1 and 'wandb' not in k and 'savedir' not in k]
    assert swept_hyps == ['seed']
    # trunk_path should exist
    assert run_configs[0].ckpt_path is not None
    trunk_config_path = os.path.join(os.path.dirname(run_configs[0].ckpt_path), 'config.json')
    assert os.path.exists(trunk_config_path)
    with open(trunk_config_path, 'r') as f:
        trunk_config_dict = json.load(f)
    # needed for saving correctly
    trunk_seed = trunk_config_dict['seed']
    split_step = pl_ckpt_path_to_step(run_configs[0].ckpt_path)
    branch_seeds = sorted([rc.seed for rc in run_configs])

    trainset, valset = get_dataset(run_configs[0])
    # subset trainset as too slow
    rng = np.random.default_rng(12345)
    ixs = rng.permutation(len(trainset)).tolist()[:1000]  # ~28000 examples total
    trainset = torch.utils.data.Subset(trainset, ixs)
    model = PlModel(run_configs[0], valset.tokenizer)

    model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    collate_fn = getattr(valset, 'collate', None)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=run_configs[0].batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=run_configs[0].batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    data_dict = evaluate_lmc(config.ckpt_paths, train_loader, val_loader, model, steps=config.num_steps)
    group_dir = os.path.dirname(os.path.dirname(config.ckpt_paths[0]))
    savedir = os.path.join(group_dir, 'lmc', f'trunk-{trunk_seed}_split-{split_step}_seed-{branch_seeds[0]}_seed-{branch_seeds[1]}')
    os.makedirs(savedir, exist_ok=True)
    np.savez(os.path.join(savedir, 'data.npz'), **data_dict)


@torch.no_grad()
def evaluate_lmc(ckpt_paths, train_loader, val_loader, model, steps=11):
    # assert that checkpoints start from same base ckpt
    assert len(ckpt_paths) == 2
    param_keys = [n for n, p in model.named_parameters() if len(p.shape) >= 2]
    state_dicts = [torch.load(cp_path, map_location='cuda:0')['state_dict'] for cp_path in ckpt_paths]

    data_dict = defaultdict(lambda: [])
    alphas = np.linspace(0, 1.0, steps)
    for alpha in tqdm(alphas):
        # set model to interpolation
        sd = {
            k: (1 - alpha) * state_dicts[0][k] + alpha * state_dicts[1][k]
            for k in state_dicts[0].keys()
        }
        # store svs/rank for averaging and plotting later
        for k in param_keys:
            P = sd[k]
            if len(P.shape) > 2:
                P = P.flatten(1)
            s = torch.linalg.svdvals(P).cpu().numpy()
            eff_rank = effective_rank_s(s, normalize=True)
            data_dict[f'{k}_sv'].append(s)
            data_dict[f'{k}_eff_rank'].append(float(eff_rank))

        for k in sd:
            model.state_dict()[k].copy_(sd[k])
        # reset batchnorm after interpolating
        if has_bn(model):
            torch.optim.swa_utils.update_bn(train_loader, model.model, next(model.parameters()).device)

        result = evaluate(train_loader, model)
        for k in result:
            data_dict[f'train/{k}'].append(float(result[k]))

        result = evaluate(val_loader, model)
        for k in result:
            data_dict[f'val/{k}'].append(float(result[k]))

        # pruned bottom
        for k in sd:
            model.state_dict()[k].copy_(sd[k])
        model = sv_prune_model_(model)
        # reset batchnorm after interpolating
        if has_bn(model):
            torch.optim.swa_utils.update_bn(train_loader, model.model, next(model.parameters()).device)

        result = evaluate(train_loader, model)
        for k in result:
            data_dict[f'train/pruned_bot_{k}'].append(float(result[k]))

        result = evaluate(val_loader, model)
        for k in result:
            data_dict[f'val/pruned_bot_{k}'].append(float(result[k]))

        # pruned top
        for k in sd:
            model.state_dict()[k].copy_(sd[k])
        model = sv_prune_model_(model, keep_bottom=True)
        # reset batchnorm after interpolating
        if has_bn(model):
            torch.optim.swa_utils.update_bn(train_loader, model.model, next(model.parameters()).device)

        result = evaluate(train_loader, model)
        for k in result:
            data_dict[f'train/pruned_top_{k}'].append(float(result[k]))

        result = evaluate(val_loader, model)
        for k in result:
            data_dict[f'val/pruned_top_{k}'].append(float(result[k]))

    # sva between endpoints
    svd_dicts = [get_svd_dict(sd, param_keys) for sd in state_dicts]
    for k in param_keys:
        sim = svd_agreement(svd_dicts[0][k], svd_dicts[1][k], diag=False)
        data_dict[f'{k}_sva'] = sim.cpu().numpy()

    data_dict = {k: np.array(v) for k, v in data_dict.items()}
    return data_dict


@torch.no_grad()
def evaluate(dataloader, pl_module):
    pl_module.eval()
    device = next(pl_module.parameters()).device
    totals = defaultdict(lambda: 0)
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='eval')):
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
        outputs = pl_module.validation_step(batch, batch_idx)
        for k in outputs.keys():
            totals[k] += float(outputs[k])

    totals['loss'] /= len(dataloader)
    totals['cer'] /= totals['num_chars']
    totals['wer'] /= totals['num_words']
    totals.pop('num_chars')
    totals.pop('num_words')

    pl_module.train()
    return totals


def get_svd_dict(state_dict, param_keys):
    svd_dict = {}
    for name in param_keys:
        P = state_dict[name]
        if len(P.shape) > 2:
            P = P.flatten(1)
        U, s, Vh = safe_svd(P)
        svd_dict[name] = (U, s, Vh)
    return svd_dict


def safe_svd(P):
    try:
        U, s, Vh = torch.linalg.svd(P, full_matrices=False)
    except:
        eps = 1e-10 * torch.eye(*P.shape, device=P.device)
        U, s, Vh = torch.linalg.svd(P + eps, full_matrices=False)
    return U, s, Vh


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


def has_lstm(module):
    for name, mod in module.named_modules():
        if isinstance(mod, torch.nn.LSTM):
            return True
    return False


def svd_agreement(svd_x, svd_y, scaled=False, diag=False, absolute=True):
    ux, sx, vhx = svd_x
    uy, sy, vhy = svd_y
    vx, vy = vhx.T, vhy.T
    assert ux.shape == uy.shape
    assert sx.shape == sy.shape
    assert vx.shape == vy.shape

    uu = ux.T.mm(uy)
    vv = vx.T.mm(vy)
    sim = uu * vv
    if scaled:
        sim *= sx[:, None] * sy[None, :]
    if absolute:
        sim = torch.abs(sim)
    if diag:
        sim = torch.diag(sim)
    return sim


def effective_rank_s(s, normalize=True):
    assert len(s.shape) == 1
    s = np.array(s)
    s += 1e-10  # add small nonzero value
    P = s / np.sum(s)
    er = np.exp(np.sum(P * -np.log(P)))
    if normalize:
        er /= len(s)
    return er


def pl_ckpt_path_to_step(ckpt_path):
    root, _ = os.path.splitext(os.path.basename(ckpt_path))
    step = int(root.split('-')[1].split('=')[1])
    return step


@torch.no_grad()
def sv_prune_model_(model, sparsity=0.5, keep_bottom=False):
    # prune to top {sparsity} * 100% of singular vectors 
    param_keys = [n for n, p in model.named_parameters() if len(p.shape) >= 2]
    # don't prune last layer
    param_keys = param_keys[:-1]

    device = next(model.parameters()).device
    for i, name in enumerate(param_keys):
        P = model.state_dict()[name]
        orig_shape = P.shape
        if len(P.shape) > 2:
            P = P.flatten(1)
        U, s, Vh = safe_svd(P)
        num_svs = int(np.ceil(sparsity * len(s)))
        if keep_bottom:
            Up, sp, Vhp = U[:, num_svs:], s[num_svs:], Vh[num_svs:, :]
        else:
            Up, sp, Vhp = U[:, :num_svs], s[:num_svs], Vh[:num_svs, :]
        P = Up.mm(torch.diag(sp)).mm(Vhp)
        P = P.reshape(orig_shape).to(device)
        model.state_dict()[name].copy_(P)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_paths', type=str, nargs='+', help='checkpoint paths for lmc')
    parser.add_argument('-n', '--num_steps', type=int, default=11, help='number of checkpoints along the linear interpolation (including endpoints)')

    # dummy args for slurm launcher
    parser.add_argument('--wandb_project', type=str, help='wandb project to log in')
    parser.add_argument('--wandb_group', type=str, help='wandb group for runs')
    parser.add_argument('--wandb_dir', type=str, help='base wandb directory')
    parser.add_argument('--wandb_name', type=str, help='wandb run id')
    config = parser.parse_args()

    assert len(config.ckpt_paths) == 2, 'need 2 checkpoints for LMC'
    config_paths = [
        os.path.join(os.path.dirname(cp_path), 'config.json')
        for cp_path in config.ckpt_paths
    ]
    run_configs = []
    for config_path in config_paths:
        assert os.path.exists(config_path), f'config path {config_path} does not exist'
        with open(config_path, 'r') as f:
            run_config_dict = json.load(f)
        run_configs.append(argparse.Namespace(**run_config_dict))

    main(config, run_configs)
