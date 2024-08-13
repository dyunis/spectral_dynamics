import argparse
from glob import glob
import itertools
import json
import os

import yaml


# - take in experiment yaml
# - get all checkpoints underneath that directory
# - create yamls for all Nc2 unique pairs of LMC jobs (same trunk ckpt, different seeds)
EXPDIR = './exps'

def main(config):
    with open(config.sweep_yaml, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    expdir = os.path.join(EXPDIR, yaml_dict['project'], yaml_dict['name'])
    ckpt_paths = yaml_dict['parameters']['ckpt_path']
    if 'value' in ckpt_paths:
        ckpt_paths = [ckpt_paths['value']]
    else:
        ckpt_paths = ckpt_paths['values']
    ckpt_paths = sorted(ckpt_paths, key=pl_ckpt_path_to_step)

    run_dirs = glob(os.path.join(expdir, f'{os.path.basename(expdir)}-*'))
    config_paths = [os.path.join(d, 'config.json') for d in run_dirs]
    config_dicts = []
    for path in config_paths:
        with open(path, 'r') as f:
            config_dicts.append(json.load(f))

    yaml_pairs = []
    for cp_path in ckpt_paths:
        # get runs launched from same ckpt
        cp_run_dirs = []
        for r, d in zip(run_dirs, config_dicts):
            if d['ckpt_path'] == cp_path:
                cp_run_dirs.append(r)

        # get last cp in each run_dir
        last_cps = []
        for d in cp_run_dirs:
            d_cps = glob(os.path.join(d, '*.ckpt'))
            d_cps = [p for p in d_cps if os.path.basename(p) != 'last.ckpt']
            d_cps = sorted(d_cps, key=pl_ckpt_path_to_step)
            last_cps.append(d_cps[-1])

        # get list of all pairs of last cps
        cp_pairs = itertools.combinations(last_cps, 2)
        cp_pairs = [list(pair) for pair in cp_pairs]
        yaml_pairs.extend(cp_pairs)

    # save yaml
    new_yaml_dict = {
        'program': 'evaluate_lmc.py',
        'command': ['${env}', '${interpreter}', '${program}'],
        'method': 'grid',
        'project': 'spectral_dynamics',
        'name': f'{yaml_dict["name"]}_lmc',
        'parameters': {
            'ckpt_paths': {'values': yaml_pairs}
        }
    }

    new_yaml_path = os.path.splitext(config.sweep_yaml)[0]
    new_yaml_path = f'{new_yaml_path}_lmc.yaml'
    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_yaml_dict, f)


def pl_ckpt_path_to_step(ckpt_path):
    root, _ = os.path.splitext(os.path.basename(ckpt_path))
    step = int(root.split('-')[1].split('=')[1])
    return step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_yaml', type=str, help='sweep base yaml')
    config = parser.parse_args()
    main(config)
