import argparse
from collections import defaultdict
from glob import glob
import itertools
import json
import os

import numpy as np
from tqdm import tqdm


# - collect all split directories
# - average metrics across all same-step splits (over trunk and pairs of branches)
# - save in new directory structure
# - also add in dim-avg metrics and total effective rank
def main(config):
    config.lmc_dir = config.lmc_dir.rstrip('/')  # remove trailing slashes
    lmc_dirs = glob(os.path.join(config.lmc_dir, 'trunk-*_trunk-*_split-*_seed-*'))
    lmc_dirs = [d for d in lmc_dirs if os.path.isdir(d)]
    split_steps = sorted(list(set([lmc_dir_to_split_step(d) for d in lmc_dirs])))

    total_mean = defaultdict(lambda: [])
    total_std = defaultdict(lambda: [])
    for split_step in tqdm(split_steps):
        split_data = defaultdict(lambda: [])
        for lmc_dir in lmc_dirs:
            if lmc_dir_to_split_step(lmc_dir) == split_step:
                # aggregate
                lmc_data = dict(np.load(os.path.join(lmc_dir, 'data.npz')))
                for k in lmc_data:
                    split_data[k].append(lmc_data[k])

        # for avg effective rank of model create giant list of all rank keys
        rank_keys = [k for k in split_data if k.endswith('eff_rank')]
        for k in rank_keys:
            split_data['eff_rank'].extend(split_data[k])

        # sva_keys -> sva_score
        sva_keys = [k for k in split_data if k.endswith('_sva') and 'dim' not in k]
        for k in sva_keys:
            scores = [inter_layer_alignment_score(al, rank=10) for al in split_data[k]]
            split_data[f'{k}_sva_score'].extend(scores)

        # for dim_avg sv(a), create giant list of all dim keys
        dims = list(set([lmc_data[k].shape[1] for k in lmc_data if k.endswith('_sv')]))
        for d in dims:
            dim_keys = [k.rstrip('sv') for k in lmc_data if k.endswith('_sv') and lmc_data[k].shape[1] == d]
            # get full list
            for ext in ('sv', 'sva', 'eff_rank'):
                for k in dim_keys:
                    split_data[f'dim_{d}_{ext}'].extend(split_data[k + ext])

        # compute LMC barriers for scalar metrics
        keys = [k for k in split_data if len(split_data[k][0].shape) == 1]
        for k in keys:
            barriers = [compute_barrier(l) for l in split_data[k]]
            split_data[f'{k}_barrier'].extend(barriers)

        # average over splits
        split_data = {k: np.array(v) for k, v in split_data.items()}
        mean_dict = {k: np.mean(v, axis=0) for k, v in split_data.items()}
        std_dict = {k: np.std(v, axis=0) for k, v in split_data.items()}
        for k in mean_dict:
            total_mean[k].append(mean_dict[k])
        for k in std_dict:
            total_std[k].append(std_dict[k])

        # save in new directory for each split step
        savedir = os.path.join(config.lmc_dir, f'split-{split_step}')
        os.makedirs(savedir, exist_ok=True)
        np.savez(os.path.join(savedir, 'mean.npz'), **mean_dict)
        np.savez(os.path.join(savedir, 'std.npz'), **std_dict)

    total_mean = {k: np.array(v) for k, v in total_mean.items()}
    total_std = {k: np.array(v) for k, v in total_std.items()}
    np.savez(os.path.join(config.lmc_dir, 'mean.npz'), **total_mean)
    np.savez(os.path.join(config.lmc_dir, 'std.npz'), **total_std)


def compute_barrier(y):
    # get maximum difference to linear interpolation (from what is being transferred in transfer learning)
    # compute sign
    linear_interp = np.linspace(y[0], y[-1], len(y))
    diff = y - linear_interp
    diff = diff[1:-1]  # don't consider endpoints
    return diff.max()


def lmc_dir_to_split_step(lmc_dir):
    base = os.path.basename(lmc_dir)
    split_step = int(base.split('_')[2].split('-')[1])
    return split_step


def inter_layer_alignment_score(alignment, rank=10):
    # take diagonal of alignment
    # sum top p fraction
    # divide by number of elements summed
    diag = np.diagonal(alignment, axis1=-2, axis2=-1)
    scores = diag[..., :rank]
    scores = scores.mean(axis=-1)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lmc_dir', type=str, help='lmc dir where directories and data are')
    config = parser.parse_args()
    main(config)
