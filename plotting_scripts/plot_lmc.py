import argparse
import os
from glob import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


matplotlib.rcParams.update({'font.size': 30, 'legend.fontsize': 20})


def main(config):
    lmc_dir = config.lmc_dir
    savedir = '-'.join(os.path.normpath(lmc_dir).split(os.sep)[-2:])
    savedir = os.path.join('plots', savedir)
    os.makedirs(savedir, exist_ok=True)
    lmc_mean_dict = dict(np.load(os.path.join(lmc_dir, 'mean.npz')))
    lmc_std_dict = dict(np.load(os.path.join(lmc_dir, 'std.npz'))) 

    # plot LMC curves for different timesteps (metrics, rank)
    # plot barrier over time
    keys = ['train/loss', 'train/pruned_bot_loss', 'train/pruned_top_loss']
    for k in keys:
        plot_barrier(savedir, lmc_mean_dict, lmc_std_dict, f'{k}_barrier')
        # steps = len(lmc_mean_dict[k])
        # for step in range(steps):
            # plot_lmc(savedir, lmc_mean_dict, lmc_std_dict, k, step)

    # plot SVA between endpoints diagonal over time
    # plot SVA between endpoints for different timesteps
    sva_keys = [k for k in lmc_mean_dict if k.endswith('sva')]
    for k in sva_keys:
        sva = lmc_mean_dict[k]
        plot_sva_diag(savedir, k, sva)
        # steps = len(sva)
        # for step in range(steps):
            # plot_sva_step(savedir, k, sva, step)

    # plot sva fingerprint (layer vs. split step)
    score_keys = [k for k in lmc_mean_dict.keys() if k.endswith('sva_score')]
    plot_sva_fingerprint(savedir, lmc_mean_dict, score_keys, 'sva_score.png')


# plot diagonal heatmap
def plot_sva_diag(savedir, key, sva, rank=100):
    sva = sva[:, :rank, :rank]
    diags = []
    for i in range(len(sva)):
        diags.append(np.diag(sva[i]))
    diags = np.array(diags)
    diags = diags.T
    # plot heatmap of sva, colorbar from 0-1
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    img = ax.imshow(diags, cmap='inferno', interpolation='nearest', vmin=0, aspect='auto')
    fig.colorbar(img, ax=ax)
    length = diags.shape[1]
    ticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
    labels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
    ax.set_xticks(ticks=ticks, labels=labels)
    ax.set_xlabel('Split epoch')
    ax.set_ylabel('Diagonal rank')
    savepath = os.path.join(savedir, 'sva', key, 'diag.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


# plot full heatmap at individual step
def plot_sva_step(savedir, key, sva, step, rank=100):
    sva = lmc_mean_dict[key]
    sva = sva[:, :rank, :rank]
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    img = ax.imshow(sva[step], cmap='inferno', interpolation='nearest')
    fig.colorbar(img, ax=ax)
    savepath = os.path.join(savedir, 'sva', key, f'step_{step}.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


# heatmap image of singvec agreement for portion of diagonal with current and final checkpoint over layers and time
def plot_sva_fingerprint(savedir, mean_dict, keys, savefile):
    if len(keys) == 0:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    mat = []
    for key in keys:
        scores = mean_dict[key]
        mat.append(scores)
    mat = np.array(mat)
    cmap = sns.color_palette('magma', as_cmap=True)
    img = ax.imshow(mat, cmap=cmap, interpolation='nearest', vmin=0, aspect='auto')
    fig.colorbar(img, ax=ax)
    length = mat.shape[-1]
    xticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
    xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
    ax.set_xticks(ticks=xticks, labels=xlabels)
    yticks = [0, mat.shape[0]-1]
    ylabels = [1, mat.shape[0]]
    ax.set_yticks(ticks=yticks, labels=ylabels)
    ax.set_xlabel('Split epoch')
    ax.set_ylabel('Layer')
    savepath = os.path.join(savedir, 'sva', savefile)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


def plot_lmc(savedir, lmc_mean_dict, lmc_std_dict, key, step, color='blue'):
    mean, std = lmc_mean_dict[key][step], lmc_std_dict[key][step]
    x = np.arange(len(mean))
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    ax.plot(x, mean, color=color)
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
    ax.set_ylabel('Alpha')
    ax.set_ylabel('Barrier')
    savepath = os.path.join(savedir, 'lmc', f'{key.replace("/", "-")}_step_{step}.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


def plot_barrier(savedir, lmc_mean_dict, lmc_std_dict, key, color='blue'):
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    mean, std = lmc_mean_dict[key], lmc_std_dict[key]
    x = np.arange(len(mean))
    ax.plot(x, mean, color=color)
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
    length = mean.shape[0]
    xticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
    xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
    ax.set_xticks(ticks=xticks, labels=xlabels)
    ax.set_xlabel('Split epoch')
    ax.set_ylabel('Barrier')
    savepath = os.path.join(savedir, f'{key.replace("/", "-")}.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lmc_dir', type=str, default=None, help='lmc directory to load mean,std data from')
    config = parser.parse_args()
    print(f'plotting {config.lmc_dir}')
    assert os.path.exists(config.lmc_dir)
    main(config)
