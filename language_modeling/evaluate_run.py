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


def main(config, run_config):
    # 'cusolver' is default, and has issues with SVDs of very tall or wide matrices
    # see https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.preferred_linalg_library
    # torch.backends.cuda.preferred_linalg_library(backend='magma')
    trainset, valset = get_dataset(run_config)
    # subset as wikitext103 is too big
    rng = np.random.default_rng(12345)
    ixs = rng.permutation(len(trainset)).tolist()[:1000]  # ~57000 examples total
    trainset = torch.utils.data.Subset(trainset, ixs)
    model = PlModel(run_config, valset.tokenizer)

    cp_paths = glob(os.path.join(config.run_dir, '*.ckpt'))
    cp_paths = [p for p in cp_paths if os.path.basename(p) != 'last.ckpt']
    cp_paths = sorted(cp_paths, key=pl_ckpt_path_to_step)
    # subsample for faster evaluation
    if config.subsample_mode == 'linear':
        cp_paths = [p for i, p in enumerate(cp_paths) if i % config.frequency == 0]
    elif config.subsample_mode == 'log':
        cp_paths = [cp_paths[i] for i in gen_log_space(len(cp_paths), len(cp_paths) // config.frequency)]

    model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    collate_fn = getattr(valset, 'collate', None)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=run_config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=run_config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    data_dict = evaluate_run(cp_paths, train_loader, val_loader, model, config.sparsity)
    np.savez(os.path.join(config.run_dir, 'data.npz'), **data_dict)


@torch.no_grad()
def evaluate_run(cp_paths, train_loader, val_loader, model, sparsity=0.5):
    param_keys = [n for n, p in model.named_parameters() if len(p.shape) >= 2]
    # assert set(pk) == set(param_keys) doesn't work for transformers because
    # of embeddings and in and out projections of multihead attention

    final_sd = torch.load(cp_paths[-1], map_location='cuda:0')['state_dict']
    final_sd_svd = {}
    for name in param_keys:
        fP = final_sd[name]
        if len(fP.shape) > 2:
            fP = fP.flatten(1)
        U, s, Vh = safe_svd(fP)
        final_sd_svd[name] = (U, s, Vh)

    data_dict = defaultdict(lambda: [])
    for path in tqdm(cp_paths):
        step = pl_ckpt_path_to_step(path)
        data_dict['step'].append(float(step))
        sd = torch.load(path, map_location='cuda:0')['state_dict']

        model.load_state_dict(sd)
        model.cuda()

        result = evaluate(train_loader, model)
        for k in result:
            data_dict[f'train/{k}'].append(float(result[k]))

        result = evaluate(val_loader, model)
        for k in result:
            data_dict[f'val/{k}'].append(float(result[k]))

        # interlayer alignment
        # for alignment don't need to decompose into multiple heads as
        # the matrices first are reshaped as q, k, v, then split to heads
        # see:
        # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/attention.cpp
        # native_multi_head_attn and qkv_projection
        pairs = model.inter_layer_pairs()
        for k1, k2 in pairs:
            p1 = sd[k1]
            p2 = sd[k2]

            # W_q to W_k^T and W_qW_k^T to W_v
            if k1 == k2 and 'in_proj_weight' in k1:
                # shapes are (d_out, d_in)
                Wq = p1[:p1.size(1)]
                Wk = p1[p1.size(1):-p1.size(1)]
                Wv = p1[-p1.size(1):]
                # split into heads at output, then SVD
                Wq = Wq.reshape(12, 64, 768)  # see model.py, 12 heads
                Wk = Wk.reshape(12, 64, 768)
                svdq = safe_svd(Wq)
                svdk = safe_svd(Wk)
                # match up output dims of Wq and Wk as those will be multiplied
                qksim = svdq[0].permute(0, 2, 1).bmm(svdk[0]).abs()
                # qksim = svdq[0].T.matmul(svdk[0]).abs()  # if not splitting heads, bad
                data_dict[f'{k1}_{k2}_qk_align'].append(qksim.cpu())
                # svdv = safe_svd(Wv)
                # svdprod = safe_svd(Wq.T.matmul(Wk))  # multiples output dims together
                # attn is (XQ)(XK)^T XV = (XQ)(K^TX^T) XV
                # now Wq shape is (d_out, d_in), so we need extra transpose
                # which gives (XWq^T)(WkX) XWv^T
                # qkv sim doesn't actually make sense as X is in the way...
                # qkvsim = svdprod[2].matmul(svdv[0]).abs()
                # data_dict[f'{k1}_{k2}_qkv_align'].append(qkvsim.cpu())
                continue

            # W_v to W_o
            if 'in_proj_weight' in k1 and 'out_proj' in k2:
                p1 = p1[-p1.size(1):]
            # W2 to W_in
            elif 'linear2.weight' in k1 and 'in_proj_weight' in k2:
                # lgtm
                pass
            elif 'linear2.weight' in k1 and k2 == 'model.decoder.weight':
                # lgtm
                pass
            _svd1 = safe_svd(p1)
            _svd2 = safe_svd(p2)
            U_prev = _svd1[0]
            Vh_next = _svd2[2]
            if k1 == 'model.encoder.weight':
                U_prev = _svd1[2].T  # embedding weight convention is reversed
            sim = Vh_next.matmul(U_prev).abs()
            data_dict[f'{k1}_{k2}_align'].append(sim.cpu())

        for name in param_keys:
            P = sd[name]
            if len(P.shape) > 2:
                P = P.flatten(1)
            U, s, Vh = safe_svd(P)
            SVD = (U, s, Vh)

            sim = svd_agreement(SVD, final_sd_svd[name], diag=False)
            # use normalized sv contribution
            eff_rank = entropy(s, normalize=True)

            data_dict[f'{name}_sv'].append(s.cpu())
            data_dict[f'{name}_sva'].append(sim.cpu())
            data_dict[f'{name}_eff_rank'].append(float(eff_rank))

        model.load_state_dict(sd)
        model = sv_prune_model_(model, sparsity)
        if has_bn(model):
            torch.optim.swa_utils.update_bn(train_loader, model.model, next(model.parameters()).device)

        result = evaluate(train_loader, model)
        for k in result:
            data_dict[f'train/pruned_bot_{k}'].append(float(result[k]))

        result = evaluate(val_loader, model)
        for k in result:
            data_dict[f'val/pruned_bot_{k}'].append(float(result[k]))

        model.load_state_dict(sd)
        model = sv_prune_model_(model, sparsity, keep_bottom=True)
        if has_bn(model):
            torch.optim.swa_utils.update_bn(train_loader, model.model, next(model.parameters()).device)

        result = evaluate(train_loader, model)
        for k in result:
            data_dict[f'train/pruned_top_{k}'].append(float(result[k]))

        result = evaluate(val_loader, model)
        for k in result:
            data_dict[f'val/pruned_top_{k}'].append(float(result[k]))

    # convert data_dict to dict of numpy arrays
    data_dict = {k: np.array(v) for k, v in data_dict.items()}
    return data_dict


@torch.no_grad()
def evaluate(dataloader, pl_module):
    pl_module.eval()
    device = next(pl_module.parameters()).device
    totals = defaultdict(lambda: 0)
    for batch_idx, batch in enumerate(dataloader):
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
        outputs = pl_module.validation_step(batch, batch_idx)
        for k in outputs.keys():
            totals[k] += float(outputs[k])
    for k in totals:
        totals[k] /= len(dataloader)
    pl_module.train()
    return totals


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


def pl_ckpt_path_to_step(ckpt_path):
    root, _ = os.path.splitext(os.path.basename(ckpt_path))
    step = int(root.split('-')[1].split('=')[1])
    return step


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


def entropy(s, normalize=True):
    assert len(s.shape) == 1
    s += 1e-10  # add small nonzero value
    probs = s / s.sum()
    ent = (-(probs * probs.log()).sum()).exp()
    if normalize:
        ent /= len(s)
    return ent


# n evenly log-spaced integers less than high
# https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
def gen_log_space(limit, n):
    result = [1]
    if n > 1:
        ratio = (float(limit)/result[-1]) ** (1.0 / (n-len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            # next value is different integer
            result.append(next_value)
        else:
            # next value is same integer, so increment
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--run_dir', type=str, help='checkpoint dir')
    parser.add_argument('-f', '--frequency', type=int, default=1, help='frequency to subsample for evaluation')
    parser.add_argument('--subsample_mode', type=str, default=None, choices=[None, 'linear', 'log'], help='linear or log-spaced subsampling for eval')
    parser.add_argument('-s', '--sparsity', type=float, default=0.5, help='sparsity to prune for prune plot')

    # dummy args for slurm launcher
    parser.add_argument('--wandb_project', type=str, help='wandb project to log in')
    parser.add_argument('--wandb_group', type=str, help='wandb group for runs')
    parser.add_argument('--wandb_dir', type=str, help='base wandb directory')
    parser.add_argument('--wandb_name', type=str, help='wandb run id')
    config = parser.parse_args()

    argpath = os.path.join(config.run_dir, 'config.json')
    assert os.path.exists(argpath), f'config path {argpath} does not exist'
    with open(argpath, 'r') as f:
        run_config_dict = json.load(f)
    run_config = argparse.Namespace(**run_config_dict)

    main(config, run_config)
