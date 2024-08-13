import argparse
from glob import glob
import os

import yaml


# - take in experiment yaml
# - get all checkpoints underneath that directory
# - create new yaml launching 3 new seeds (3, 4, 5) from all of the checkpoints previously generated
EXPDIR = './exps'

def main(config):
    with open(config.sweep_yaml, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    expdir = os.path.join(EXPDIR, yaml_dict['project'], yaml_dict['name'])
    seeds = yaml_dict['parameters']['seed']['values']
    expdirs = [os.path.join(expdir, f'{os.path.basename(expdir)}-{seed}') for seed in seeds]
    ckpt_paths = []
    for dirpath in expdirs:
        cp_paths = get_and_trim_cp_paths(dirpath, config)
        ckpt_paths.extend(cp_paths)

    # new seeds for branches
    new_seeds = [max(seeds) + 1 + i for i in range(config.num_branches)]
    yaml_dict['parameters']['seed'] = {'values': new_seeds}
    yaml_dict['parameters']['ckpt_path'] = {'values': ckpt_paths}
    yaml_dict['name'] = yaml_dict['name'] + '_branches'

    new_yaml_path = os.path.splitext(config.sweep_yaml)[0]
    new_yaml_path = f'{new_yaml_path}_branches.yaml'
    with open(new_yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f)


def get_and_trim_cp_paths(expdir, config):
    # get all checkpoints and sort
    cp_paths = glob(os.path.join(expdir, '*.ckpt'))
    cp_paths = [p for p in cp_paths if os.path.basename(p) != 'last.ckpt']
    cp_paths = sorted(cp_paths, key=pl_ckpt_path_to_step)

    # trim cp_paths down
    total_splits = len(cp_paths)
    if config.num_even_splits != -1:
        if config.num_even_splits < total_splits:
            frequency = total_splits // config.num_even_splits
            cp_paths = [p for i, p in enumerate(cp_paths) if i % frequency == 0 and i != len(cp_paths) - 1]
        assert len(cp_paths) > 0
        # assert len(cp_paths) == num_splits + 1
        for path in cp_paths:
            assert os.path.exists(path)
    if config.num_early_splits != -1:
        cp_paths = cp_paths[:config.num_early_splits]
    return cp_paths


def pl_ckpt_path_to_step(ckpt_path):
    root, _ = os.path.splitext(os.path.basename(ckpt_path))
    step = int(root.split('-')[1].split('=')[1])
    return step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_yaml', type=str, help='sweep base yaml')
    parser.add_argument('-b', '--num_branches', type=int, default=3, help='number of branches to split at each point')
    parser.add_argument('--num_even_splits', type=int, default=-1, help='keep this many evenly spaced checkpoints')
    parser.add_argument('--num_early_splits', type=int, default=-1, help='keep this many checkpoints from the beginning, can be combined with above')
    config = parser.parse_args()
    main(config)
