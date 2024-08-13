import argparse
from copy import deepcopy
from glob import glob
import os

import yaml


# - take in experiment yaml
# - get all subdirs
# - launch evaluate job for each one
EXPDIR = './exps'

def main(config):
    with open(config.sweep_yaml, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    expdir = os.path.join(EXPDIR, yaml_dict['project'], yaml_dict['name'])
    seeds = yaml_dict['parameters']['seed']['values']
    expdirs = sorted(list(glob(os.path.join(expdir, f'{os.path.basename(expdir)}-*'))), key=lambda p: int(p.split('-')[1]))

    # construct yaml_dict for evaluation
    new_yaml_dict = deepcopy(yaml_dict)
    new_yaml_dict['program'] = os.path.join(os.path.dirname(yaml_dict['name']), 'evaluate_run.py')
    new_yaml_dict['name'] = yaml_dict['name'] + '_eval'
    new_yaml_dict['parameters'] = {}
    new_yaml_dict['parameters']['run_dir'] = {'values': expdirs}
    new_yaml_dict['parameters']['frequency'] = {'value': config.frequency}

    new_yaml_path = os.path.splitext(config.sweep_yaml)[0]
    new_yaml_path = f'{new_yaml_path}_eval.yaml'
    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_yaml_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_yaml', type=str, help='sweep base yaml')
    parser.add_argument('-f', '--frequency', type=int, default=1, help='frequency to subsample for evaluation')
    config = parser.parse_args()
    main(config)
