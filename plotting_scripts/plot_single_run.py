import argparse
import os
from glob import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def main(metrics_dir):
    metrics_dir = config.metrics_dir
    if 'modadd' in metrics_dir or 'rethinking' in metrics_dir:
        matplotlib.rcParams.update({'font.size': 20, 'legend.fontsize': 16})
    else:
        matplotlib.rcParams.update({'font.size': 30, 'legend.fontsize': 20})
    matplotlib.rcParams.update({'font.size': 20, 'legend.fontsize': 16})
    savedir = '-'.join(os.path.normpath(metrics_dir).split(os.sep)[-2:])
    savedir = os.path.join('plots', savedir)
    os.makedirs(savedir, exist_ok=True)
    data_dict = dict(np.load(os.path.join(metrics_dir, 'data.npz')))

    compute_extra_metrics_(data_dict)
    plot_data(savedir, data_dict)


def compute_extra_metrics(data_dict):
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


def plot_data(savedir, data_dict)
    mean_dict
    std_dict = {k: np.ones_like(v) for k, v in mean_dict.items()}

    splits = ['train', 'val']
    metric_keys = ['loss', 'err', 'cer', 'wer']
    labels = ['Loss', 'Err.', 'CER', 'WER']
    for split in splits:
        for ix, key in enumerate(metric_keys):
            if f'{split}/{key}' not in mean_dict:
                continue
            plot_keys = [f'{split}/{key}', f'{split}/pruned_bot_{key}', f'{split}/pruned_top_{key}']
            colors = matplotlib.colormaps['Set1'].colors
            fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
            for k, color in zip(plot_keys, colors):
                plot_mean_std(savedir, mean_dict, std_dict, k, ax=ax, color=color)
            ax.legend()
            length = len(mean_dict[k])
            ticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
            xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
            # xlabels = ['', 1, 20000, 40000]  # grokking
            # xlabels = ['', 1, 50000, 100000]  # slingshot grokking
            ax.set_xticks(ticks=ticks, labels=xlabels)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(labels[ix])
            fig.savefig(os.path.join(savedir, f'{split}-{key}.png'), bbox_inches='tight')
            plt.close(fig)

    # only for grokking plots
    if 'modadd' in savedir:
        plot_keys = ['train/err', 'val/err']
        colors = matplotlib.colormaps['Set1'].colors
        fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
        for k, color in zip(plot_keys, colors):
            plot_mean_std(savedir, mean_dict, std_dict, k, ax=ax, color=color)
        ax.legend()
        length = len(mean_dict[k])
        ticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
        xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
        # xlabels = ['', 1, 20000, 40000]  # grokking
        # xlabels = ['', 1, 50000, 100000]  # slingshot grokking
        ax.set_xticks(ticks=ticks, labels=xlabels)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Err.')
        fig.savefig(os.path.join(savedir, f'error.png'), bbox_inches='tight')
        plt.close(fig)

    # plot of svs for each layer
    sv_keys = [k for k in mean_dict if k.endswith('sv')]
    for k in sv_keys:
        plot_svs(savedir, mean_dict, k)

    gini_keys = [k for k in mean_dict.keys() if k.endswith('sv_gini')]
    plot_gini_fingerprint(savedir, mean_dict, gini_keys)

    rank_keys = [k for k in mean_dict if k.endswith('eff_rank') and ('model' in k or 'gpt_neox' in k)]
    plot_rank_fingerprint(savedir, mean_dict, rank_keys)

    # plot of sva for each layer over time
    sva_keys = [k for k in mean_dict if k.endswith('sva')]
    for k in sva_keys:
        sva = mean_dict[k]
        plot_sva_diag(savedir, k, sva)

    # will be skipped if not masking exp
    sva_unmask_keys = [k for k in mean_dict if k.endswith('sva_unmask')]
    for k in sva_unmask_keys:
        sva = mean_dict[k]
        plot_sva_diag(savedir, k, sva)

    # will be skipped if not masking exp
    sva_mask_final_keys = [k for k in mean_dict if k.endswith('sva_mask_final')]
    for k in sva_mask_final_keys:
        sva = mean_dict[k]
        plot_sva_step(savedir, k, sva, step=0)

    # get alignment score for every layer
    # get average alignment score across all layers
    align_keys = [k for k in data_dict if k.endswith('align')]
    align_shapes = list(set([c_dict[k].shape[-2:] for k in align_keys]))
    for k in align_keys:
        scores = [inter_layer_alignment_score(al, rank=10) for al in data_dict[k]]
        data_dict[f'{k}_align_score'].extend(scores)

    score_keys = [k for k in mean_dict.keys() if k.endswith('sva_score')]
    plot_sva_fingerprint(savedir, mean_dict, score_keys, 'sva_score.png')

    # skipped if not masking exp
    score_keys = [k for k in mean_dict.keys() if k.endswith('sva_unmask_score')]
    plot_sva_fingerprint(savedir, mean_dict, score_keys, 'sva_unmask_score.png')

    # skiped if not masking exp
    score_keys = [k for k in mean_dict.keys() if k.endswith('sva_mask_final_score')]
    plot_sva_mask_final_fingerprint(savedir, mean_dict, score_keys)

    # plot inter-layer alignment
    align_keys = [k for k in mean_dict.keys() if k.endswith('align')]
    # align_score_keys = [k for k in mean_dict.keys() if k.endswith('align_score')]

    for k in align_keys:
        sva = mean_dict[k]
        if len(sva.shape) == 5:
            # for conv layers we have spatial dimension
            sva = sva[:, sva.shape[1]//2, sva.shape[2]//2]  # center position
            # sva = np.mean(sva, axis=(1, 2))  # mean over spatial positions
            # sva = sva[:, 0, 0]  # corner
            # sva = sva[:, 0, 2]  # corner
        elif len(sva.shape) == 4 and 'lstm' in savedir:
            # for lstm layers we have 4 separate matrices in each param
            # gate mat has cleanest signal
            # sva = sva[:, 0]  # input mat
            # sva = sva[:, 1]  # forget mat
            sva = sva[:, 2]  # gate mat
            # sva = sva[:, 3]  # output mat
        elif len(sva.shape) == 4 and ('tfmr' in savedir or 'pythia' in savedir):
            # attention layers with multiple heads, take average over heads
            sva = sva.mean(axis=1)
        plot_align_diag(savedir, k, sva, rank=100)

    # "fingerprint" bar charts for different alignment scores
    score_keys = [k for k in mean_dict.keys() if k.endswith('align_score')]
    plot_align_fingerprint(savedir, mean_dict, score_keys)


# plot of eval metrics
def plot_mean_std(savedir, mean_dict, std_dict, key, ax=None, color='blue', ylabel=None):
    save = False
    if ax is None:
        save = True
        fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    mean, std = mean_dict[key], std_dict[key]
    x = np.arange(len(mean))
    ax.plot(x, mean, color=color, label=key, linewidth=4)
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
    if save:
        length = len(x)
        ticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
        xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
        # xlabels = ['', 1, 20000, 40000]  # grokking
        # xlabels = ['', 1, 50000, 100000]  # slingshot grokking
        ax.set_xticks(ticks=ticks, labels=xlabels)
        ax.set_xlabel('Epoch')
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        savepath = os.path.join(savedir, 'metrics', f'{key.replace("/", "-")}.png')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)


def plot_svs(savedir, mean_dict, key):
    svs = mean_dict[key]
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    num_svs = svs.shape[1]
    colormap = matplotlib.colormaps['flare_r'].resampled(num_svs)
    for i in range(num_svs):
        ax.plot(svs[:, i], color=colormap(i))
    length = svs.shape[0]
    ticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
    xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
    # xlabels = ['', 1, 20000, 40000]  # grokking
    # xlabels = ['', 1, 50000, 100000]  # slingshot grokking
    ax.set_xticks(ticks=ticks, labels=xlabels)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SV')
    savepath = os.path.join(savedir, 'sv', f'{key.replace("/", "-")}.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


# heatmap of gini index of singval distribution over layers and time
def plot_gini_fingerprint(savedir, mean_dict, keys):
    if len(keys) == 0:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    mat = []
    for key in keys:
        scores = mean_dict[key]
        mat.append(scores)
    mat = np.array(mat)
    cmap = sns.color_palette('rocket', as_cmap=True)
    # cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, reverse=True)
    img = ax.imshow(mat, cmap=cmap, interpolation='nearest', vmin=0, aspect='auto')
    length = mat.shape[1]
    ticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
    xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
    # xlabels = ['', 1, 20000, 40000]  # grokking
    # xlabels = ['', 1, 50000, 100000]  # slingshot grokking
    ax.set_xticks(ticks=ticks, labels=xlabels)
    yticks = [0, mat.shape[0]-1]
    ylabels = [1, mat.shape[0]]
    ax.set_yticks(ticks=yticks, labels=ylabels)
    fig.colorbar(img, ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Layer')
    savepath = os.path.join(savedir, 'sv', f'gini.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


# heatmap of eff rank of matrices over layers and time
def plot_rank_fingerprint(savedir, mean_dict, keys):
    if len(keys) == 0:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    mat = []
    for key in keys:
        scores = mean_dict[key]
        mat.append(scores)
    mat = np.array(mat)
    cmap = sns.color_palette('rocket', as_cmap=True)
    # cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True, reverse=True)
    img = ax.imshow(mat, cmap=cmap, interpolation='nearest', vmax=1.0, aspect='auto')
    length = mat.shape[1]
    xticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
    xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
    # xlabels = ['', 1, 20000, 40000]  # grokking
    # xlabels = ['', 1, 50000, 100000]  # slingshot grokking
    ax.set_xticks(ticks=xticks, labels=xlabels)
    yticks = [0, mat.shape[0]-1]
    ylabels = [1, mat.shape[0]]
    ax.set_yticks(ticks=yticks, labels=ylabels)
    fig.colorbar(img, ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Layer')
    savepath = os.path.join(savedir, 'sv', f'eff_rank.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


# plot full heatmap at individual step
def plot_sva_step(savedir, key, sva, step, rank=100):
    sva = sva[:, :rank, :rank]
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    img = ax.imshow(sva[step], cmap='inferno', interpolation='nearest', vmin=0, aspect='auto')
    fig.colorbar(img, ax=ax)
    ax.set_xlabel('Rank i')
    ax.set_ylabel('Rank j')
    savepath = os.path.join(savedir, 'sva', key, f'step_{step}.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


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
    xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
    # xlabels = ['', 1, 20000, 40000]  # grokking
    # xlabels = ['', 1, 50000, 100000]  # slingshot grokking
    ax.set_xticks(ticks=ticks, labels=xlabels)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Diagonal rank')
    savepath = os.path.join(savedir, 'sva', key, 'diag.png')
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
    ticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
    xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
    # xlabels = ['', 1, 20000, 40000]  # grokking
    # xlabels = ['', 1, 50000, 100000]  # slingshot grokking
    ax.set_xticks(ticks=ticks, labels=xlabels)
    yticks = [0, mat.shape[0]-1]
    ylabels = [1, mat.shape[0]]
    ax.set_yticks(ticks=yticks, labels=ylabels)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Layer')
    savepath = os.path.join(savedir, 'sva', savefile)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


# bar chart of singvec agreement of portion of diagonal between unmasked and mask final checkpoint over layers
def plot_sva_mask_final_fingerprint(savedir, mean_dict, keys):
    if len(keys) == 0:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    mat = []
    for key in keys:
        scores = mean_dict[key]
        mat.append(scores)
    mat = np.array(mat).ravel()
    cmap = matplotlib.colormaps['flare_r'].resampled(mat.shape[0])
    colors = [cmap(i) for i in range(len(mat))]
    ax.bar(np.arange(len(mat)), mat, color=colors)
    ax.set_xlabel('Layer pair')
    savepath = os.path.join(savedir, 'sva', 'sva_mask_final_score.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


# plot full heatmap at individual step
def plot_align_step(savedir, key, sva, step, rank=100):
    sva = sva[:, :rank, :rank]
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    img = ax.imshow(sva[step], cmap='viridis', interpolation='nearest', vmin=0, aspect='auto')
    fig.colorbar(img, ax=ax)
    ax.set_xlabel('Rank i')
    ax.set_xlabel('Rank j')
    savepath = os.path.join(savedir, 'align', key, f'step_{step}.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


# plot diagonal heatmap
def plot_align_diag(savedir, key, sva, rank=100):
    sva = sva[:, :rank, :rank]
    diags = []
    for i in range(len(sva)):
        diags.append(np.diag(sva[i]))
    diags = np.array(diags)
    diags = diags.T
    # plot heatmap of sva, colorbar from 0-1
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    img = ax.imshow(diags, cmap='viridis', interpolation='nearest', vmin=0, aspect='auto')
    fig.colorbar(img, ax=ax)
    length = diags.shape[1]
    ticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
    xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
    # xlabels = ['', 1, 20000, 40000]  # grokking
    # xlabels = ['', 1, 50000, 100000]  # slingshot grokking
    ax.set_xticks(ticks=ticks, labels=xlabels)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Diagonal rank')
    savepath = os.path.join(savedir, 'align', key, 'diag.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


# heatmap image of consecutive layer pair singvec agreement for portion of the diagonal over the course of training
def plot_align_fingerprint(savedir, mean_dict, keys):
    if len(keys) == 0:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.8), layout='constrained')
    mat = []
    for key in keys:
        align = mean_dict[key]
        if len(align.shape) == 3 and 'conv' in key:
            # middle position for conv kernel
            align = align[:, align.shape[1]//2, align.shape[2]//2]
        elif len(align.shape) == 2 and ('tfmr' in savedir or 'pythia' in savedir):
            # average over heads for transformer
            align = align.mean(axis=1)
        elif len(align.shape) == 2 and 'lstm' in savedir:
            # output mat for lstm layers
            align = align[:, 2]
        mat.append(align)
    mat = np.array(mat)
    # cmap = sns.cubehelix_palette(start=0.5, rot=-0.75, dark=0.0, light=1.0, reverse=True, as_cmap=True)
    cmap = sns.color_palette('mako', as_cmap=True)
    img = ax.imshow(mat, cmap=cmap, interpolation='nearest', vmin=0, aspect='auto')
    fig.colorbar(img, ax=ax)
    length = mat.shape[1]
    ticks = [0, 5, 5 + (length-1-5)//2, length-1] # two more ticks halfway and full
    xlabels = [0, 1, (length-5)//2, length-5] # evenly spaced til end
    # xlabels = ['', 1, 20000, 40000]  # grokking
    # xlabels = ['', 1, 50000, 100000]  # slingshot grokking
    ax.set_xticks(ticks=ticks, labels=xlabels)
    yticks = [0, mat.shape[0]-1]
    ylabels = [1, mat.shape[0]]
    ax.set_yticks(ticks=yticks, labels=ylabels)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Layer pair')
    savepath = os.path.join(savedir, 'align', f'align_score.png')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)


def inter_layer_alignment_score(alignment, rank=10):
    # take diagonal of alignment
    # sum top p fraction
    # divide by number of elements summed
    diag = np.diagonal(alignment, axis1=-2, axis2=-1)
    scores = diag[..., :rank]
    scores = scores.mean(axis=-1)
    return scores


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
    parser.add_argument('metrics_dir', type=str, default=None, help='metrics directory to load mean,std data from')
    config = parser.parse_args()
    print(f'plotting {config.metrics_dir}')
    assert os.path.exists(config.metrics_dir)
    main(config)
