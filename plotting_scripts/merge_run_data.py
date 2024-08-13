import argparse
from collections import defaultdict
import gc
from glob import glob
import itertools
import json
import os
import pickle

import numpy as np


# compute avg run plots over seeds for different hyp combos run
# reading data from all of {sweep_name}/{run_name}/{plots}
# so save as {sweep_name}/{hyp_name}_{value}/{avg plots over seed}
# need to read in all csvs and config dicts, group runs, then average
# values in the config dicts and remake the plots saving in the new directory
# there may be a mismatch in timesteps for different runs, so average where
# they match, and take the union where they don't
def main(config):
    config.group_dir = config.group_dir.rstrip('/')  # remove trailing slashes
    group_name = os.path.basename(config.group_dir)
    config_paths = glob(os.path.join(config.group_dir, f'{group_name}-*', 'config.json'))
    npz_paths = glob(os.path.join(config.group_dir, f'{group_name}-*', 'data.npz'))
    config_paths = sorted(config_paths)
    npz_paths = sorted(npz_paths)
    assert len(config_paths) > 0, f'no config paths globbed in {config.group_dir}'
    assert len(config_paths) == len(npz_paths), f'configs {config_paths} do not correspond to npzs {npz_paths}'
    for p1, p2 in zip(config_paths, npz_paths):
        run_name1 = os.path.basename(os.path.dirname(p1))
        run_name2 = os.path.basename(os.path.dirname(p2))
        assert run_name1 == run_name2, f'run names {run_name1} and {run_name2} are different'

    # group runs by whether they differ only by a seed
    config_dicts = []
    for p in config_paths:
        with open(p, 'r') as f:
            config_dicts.append(json.load(f))

    # get all unique hyp values that aren't seed, assumes sweep is over one hyp besides seed
    all_hyps = {k: list(set([config_dict[k] for config_dict in config_dicts])) for k in config_dicts[0]}
    swept_hyps = [k for k in all_hyps if len(all_hyps[k]) > 1 and 'seed' not in k and 'wandb' not in k and 'savedir' not in k]
    # if masking exp, sweep over reset ckpts from different seeds
    if 'mask_sparsity' in swept_hyps:
        swept_hyps.remove('ckpt_path')
    swept_hyps = sorted(swept_hyps)
    if len(swept_hyps) > 0:
        # sweep hyps besides seed
        hyp_combos = list(itertools.product(*[all_hyps[k] for k in swept_hyps]))
        hyp_combos = [{k: v for k, v in zip(swept_hyps, combo)} for combo in hyp_combos]
    else:
        # no non-seed hyps swept, just pick some constant hyp
        hyp_combos = [{'lr': all_hyps['lr'][0]}]

    # for every hyp combo
    hyp_dict = {
        'rank': {},
        'train/loss': {},
        'val/loss': {}
    }
    for combo in hyp_combos:
        # encapsulate so memory gets freed
        merge_combo_metrics(config, combo, config_dicts, npz_paths, config_paths, hyp_dict)
        gc.collect()

    # save hyp_dict json under group_dir
    with open(os.path.join(config.group_dir, 'hyp_data.json'), 'w') as f:
        json.dump(hyp_dict, f, indent=2)


def merge_combo_metrics(config, combo, config_dicts, npz_paths, config_paths, hyp_dict):
    # find all configs matching that combo
    combo_npzs = []
    for d, p, cp in zip(config_dicts, npz_paths, config_paths):
        use_config = True
        for c in combo: 
            if combo[c] != d[c]:
                use_config = False
                break
        if use_config:
            combo_npzs.append(p)
            config_path = cp
        else:
            continue

    if len(combo_npzs) <= 0:
        return

    # collect all data matching hyp combo
    data_dict = defaultdict(list)
    for p in combo_npzs:
        # read in respective npzs matching config
        c_dict = dict(np.load(p))
        for k in c_dict:
            data_dict[k].append(c_dict[k])

    # for avg effective rank of model create giant list of all rank keys
    rank_keys = [k for k in data_dict if k.endswith('eff_rank')]
    for k in rank_keys:
        data_dict['eff_rank'].extend(data_dict[k])

    rank_keys = [k for k in data_dict if k.endswith('eff_rank_ratio')]
    for k in rank_keys:
        data_dict['eff_rank_ratio'].extend(data_dict[k])

    rank_keys = [k for k in data_dict if k.endswith('eff_rank_count')]
    for k in rank_keys:
        data_dict['eff_rank_count'].extend(data_dict[k])

    sv_keys = [k for k in c_dict if k.endswith('_sv') and 'dim' not in k]
    for k in sv_keys:
        arr = np.array(data_dict[k])
        arrgini = gini(arr, axis=2)
        data_dict[k.rstrip('_sv') + '_sv_gini'].extend(arrgini.tolist())

    sva_keys = [k for k in data_dict if k.endswith('sva') and 'dim' not in k]
    for k in sva_keys:
        scores = [inter_layer_alignment_score(al, rank=10) for al in data_dict[k]]
        data_dict[f'{k}_sva_score'].extend(scores)

    sva_unmask_keys = [k for k in data_dict if k.endswith('sva_unmask') and 'dim' not in k]
    for k in sva_unmask_keys:
        scores = [inter_layer_alignment_score(al, rank=10) for al in data_dict[k]]
        data_dict[f'{k}_sva_unmask_score'].extend(scores)

    sva_mask_final_keys = [k for k in data_dict if k.endswith('sva_mask_final') and 'dim' not in k]
    for k in sva_mask_final_keys:
        scores = [inter_layer_alignment_score(al, rank=10) for al in data_dict[k]]
        data_dict[f'{k}_sva_mask_final_score'].extend(scores)

    # for dim_avg sv(a), create giant list of all dim keys
    dims = list(set([c_dict[k].shape[1] for k in c_dict if k.endswith('_sv')]))
    for d in dims:
        dim_keys = [k.rstrip('sv') for k in c_dict if k.endswith('_sv') and c_dict[k].shape[1] == d]
        # get full list
        for ext in ('sv', 'sva', 'sva_unmask', 'sva_mask_final', 'eff_rank', 'eff_rank_ratio', 'eff_rank_count'):
            for k in dim_keys:
                if k + ext not in data_dict:
                    continue
                data_dict[f'dim_{d}_{ext}'].extend(data_dict[k + ext])

    # get alignment score for every layer
    # get average alignment score across all layers
    align_keys = [k for k in data_dict if k.endswith('align')]
    align_shapes = list(set([c_dict[k].shape[-2:] for k in align_keys]))
    for k in align_keys:
        scores = [inter_layer_alignment_score(al, rank=10) for al in data_dict[k]]
        data_dict[f'{k}_align_score'].extend(scores)

    for shape in align_shapes:
        shape_keys = [k for k in align_keys if c_dict[k].shape[-2:] == shape]
        shape_str = f'{shape[0]}_{shape[1]}'
        for k in shape_keys:
            data_dict[f'shape_{shape_str}_align'].extend(data_dict[k])

    data_dict = {k: np.array(v) for k, v in data_dict.items()}
    mean_dict = {k: np.mean(v, axis=0) for k, v in data_dict.items()}
    std_dict = {k: np.std(v, axis=0) for k, v in data_dict.items()}

    # turn combo into string name
    combo_name = [f'{k}_{v}' for k, v in combo.items()]
    combo_name = '-'.join(combo_name)
    savedir = os.path.join(config.group_dir, combo_name)
    os.makedirs(savedir, exist_ok=True)
    # save mean/std dict for hyp under new hyp dir
    np.savez(os.path.join(savedir, 'mean.npz'), **mean_dict)
    np.savez(os.path.join(savedir, 'std.npz'), **std_dict)

    rank_avg, rank_std = mean_dict['eff_rank'], std_dict['eff_rank']

    vloss_avg, vloss_std = mean_dict['val/loss'], std_dict['val/loss']
    best_epoch = np.argmin(vloss_avg)
    hyp_dict['rank'][combo_name] = (float(rank_avg[best_epoch]), float(rank_std[best_epoch]))
    hyp_dict['train/loss'][combo_name] = (float(mean_dict['train/loss'][best_epoch]), float(std_dict['train/loss'][best_epoch]))
    hyp_dict['val/loss'][combo_name] = (float(mean_dict['val/loss'][best_epoch]), float(std_dict['val/loss'][best_epoch]))


def inter_layer_alignment_score(alignment, rank=10):
    # take diagonal of alignment
    # sum top p fraction
    # divide by number of elements summed
    diag = np.diagonal(alignment, axis1=-2, axis2=-1)
    scores = diag[..., :rank]
    scores = scores.mean(axis=-1)
    return scores


def skewness(arr, axis=None):
    # based on third moment, see https://en.wikipedia.org/wiki/Skewness
    if axis is None:
        arr = arr.reshape(-1)
        axis = 0
    mean = arr.mean(axis=axis, keepdims=True)
    num = ((arr - mean) ** 3).mean(axis=axis, keepdims=True)
    denom = (((arr - mean) ** 2).mean(axis=axis, keepdims=True)) ** 1.5
    skew = num / denom
    return num / denom


# calculates the gini coefficient along the given axis. For example, if arr.shape = (3,100, 4), gini(arr,1).shape = (3,4)
def gini(arr, axis=None):
    if axis is None:
        arr = arr.reshape(-1)
        axis = 0
    mad = np.zeros(shape = [arr.shape[i] for i in range(len(arr.shape)) if i != axis])
    slc = [slice(None)] * len(arr.shape)
    slice_shape = list(arr.shape)
    slice_shape[axis] = 1
    for i in range(arr.shape[axis]):
        slc[axis] = i
        mad += abs(arr - arr[tuple(slc)].reshape(slice_shape)).sum(axis = axis)
    return mad / (2*arr.shape[axis]**2 * arr.mean(axis = axis))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('group_dir', type=str, help='group dir of experiments that were run and evaluated')
    config = parser.parse_args()
    main(config)
