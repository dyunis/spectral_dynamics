import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess

import yaml

def main(cfg):
    # in: yaml file for sweep, [other args for sbatch]
    # check yaml file for important args
    with open(cfg.sweep_yaml, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    keys = list(yaml_dict['parameters'].keys())
    exps = get_exps(yaml_dict)
    exps = [{k: v for k, v in zip(keys, e)} for e in exps]
    wandb_dir = cfg.wandb_dir
    wandb_project = yaml_dict['project']
    wandb_group = yaml_dict['name']
    wandb_program = os.path.join(cfg.working_dir, yaml_dict['program'])

    # check for pdb calls in wandb_program to not hog resources
    with open(wandb_program, 'r') as f:
        for i, line in enumerate(f):
            if 'pdb' in line:
                raise ValueError(f'Pdb call in line {i} of {wandb_program}: {line}')

    # make_sbatch_script.py -w [project] -s [sweep id] [other args]
    sbatch_cfg_dict = {k: v for k, v in vars(cfg).items() if k not in ('sweep_yaml', 'num_jobs')}
    sbatch_cfg_dict = {k: v for k, v in sbatch_cfg_dict.items() if v is not None}
    # save script paths ahead of time based on exps, check if they exist
    sbatch_scripts = []

    # name based on hyps swept
    for i, exp in enumerate(exps):
        # if script doesn't exist, make it
        # dhash = hashlib.md5()
        # encoded = json.dumps(exp, sort_keys=True).encode()
        # name = hashlib.md5(encoded).hexdigest()[:8]
        name = f'{wandb_group}-{i}'
        exp.update({'wandb_project': wandb_project, 'wandb_group': wandb_group, 'wandb_name': name, 'wandb_dir': wandb_dir})
        # hash names here, provide wandb_dir and name as train script args
        # create sbatch script
        path = make_sbatch_script(wandb_dir, wandb_project, wandb_group, name, wandb_program, exp, sbatch_cfg_dict)
        sbatch_scripts.append(path)

    assert len(sbatch_scripts) == len(exps)

    # call sbatch on the script, show error if it doesn't work
    for script_path in sbatch_scripts:
        for i in range(cfg.num_jobs):
            out, err = run_cmd(['sbatch', script_path])
            print(out)


def get_exps(yaml_dict):
    assert yaml_dict['method'] == 'grid'
    # get cartesian product of all values in parameters
    param_val_lists = []
    for k in yaml_dict['parameters']:
        keys = list(yaml_dict['parameters'][k].keys())
        assert len(keys) == 1
        if keys[0] == 'value':
            param_val_lists.append([yaml_dict['parameters'][k]['value']])
        elif keys[0] == 'values':
            param_val_lists.append(yaml_dict['parameters'][k]['values'])
        else:
            raise ValueError(f'YAML file parameter {k} has something other than value or values as key')
    exps = list(itertools.product(*param_val_lists))
    return exps


def exp2args(exp):
    lines = []
    for k, v in exp.items():
        if v is None:
            continue
        elif isinstance(v, list):
            vals = ' '.join(v)
            line = ' '.join([f'--{k}', vals])
        elif isinstance(v, bool):
            if v:
                line = f'--{k}'
            else:
                continue
        else:
            line = f'--{k}={v}'
        lines.append(line)
    args = ' '.join(lines)
    return args


def run_cmd(command):
    out = subprocess.run(command, check=True, capture_output=True)  
    stdout = out.stdout.decode('utf-8')
    stderr = out.stderr.decode('utf-8')
    if out.returncode != 0:
        raise RuntimeError(f'Command {command} failed with {stderr}')
    return stdout, stderr


def make_sbatch_script(wandb_dir, wandb_project, wandb_group, wandb_name, wandb_program, run_args, sbatch_args):
    savedir = os.path.join(wandb_dir, wandb_project, wandb_group, wandb_name, 'slurm')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    sbatch_path = os.path.join(savedir, 'sbatch_script.sh')
    stump_path = os.path.join(savedir, 'stump_script.sh')
    
    num_cores = sbatch_args['gpus']
    if 'num_gpus' in run_args:
        print(f'Overriding specified number of cores with yaml file num_gpus: {run_args["num_gpus"]}')
        num_cores = run_args['num_gpus']

    # see https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
    lines = [
        '#!/bin/bash',
        f'#SBATCH --dependency={sbatch_args["dependency"]}',
        f'#SBATCH --partition={sbatch_args["partition"]}',
        f'#SBATCH --gpus={num_cores}',
        f'#SBATCH --ntasks-per-node=1',  # supposedly needs to match for lightning, except we launch the processes ourself (LightningEnvironment)
        f'#SBATCH --nodes=1',  # our cluster doesn't support multi-node training
    ]

    if cfg.exclude is not None:
        lines.append(
            f'#SBATCH --exclude={sbatch_args["exclude"]}',
        )

    if cfg.constraint is not None:
        lines.append(
            f'#SBATCH --constraint={sbatch_args["constraint"]}',
        )

    if cfg.array is not None:
        lines.append(
            f'#SBATCH --array=0-{sbatch_args["array"] - 1}',
        )

    lines.extend([
        f'#SBATCH --mail-user={sbatch_args["mail_user"]}',
        f'#SBATCH --mail-type={sbatch_args["mail_type"]}',
        f'#SBATCH --job-name={run_args["wandb_name"]}',
        f'#SBATCH --output={savedir}/%j.out',
        f'#SBATCH --error={savedir}/%j.err',
        '',
        'env',
        '/usr/local/bin/gpulockcheck.sh',
        '',
        f'apptainer exec --nv {run_args["container_path"]} bash {stump_path}',
    ])

    with open(sbatch_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    wandb_program_args = exp2args(run_args)
    lines = [
        '#!/bin/bash',
        'source /conda_tmp/mc3/bin/activate',
        f'cd {sbatch_args["working_dir"]}',
        'wandb login',
        f'python {wandb_program} {wandb_program_args}',
    ]
    with open(stump_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    return sbatch_path


def add_args(parser):
    parser.add_argument('sweep_yaml', type=str)
    parser.add_argument('-n', '--num_jobs', type=int, default=1, help='number of times to run singleton job')
    parser.add_argument('-d', '--dependency', type=str, default='singleton')
    parser.add_argument('-x', '--exclude', default='', type=str)  # gpu7 fused adam issue, gpu16 bf16 issue
    parser.add_argument('-p', '--partition', type=str, default=None)
    parser.add_argument('-G', '--gpus', type=int, default=1)
    parser.add_argument('-C', '--constraint', default='', type=str)
    parser.add_argument('--mail_user', type=str, default=None)
    parser.add_argument('--mail_type', type=str, default='FAIL')
    parser.add_argument('-a', '--array', type=int, help='number of array jobs to run in parallel')
    parser.add_argument('--wandb_dir', type=str, default='./exps', help='default wandb directory')
    parser.add_argument('--working_dir', type=str, default='.', help='default working directory')
    parser.add_argument('--container_path', type=str, default=None, help='path to apptainer container .sif file')
    return parser


def check_args(cfg):
    assert os.path.exists(cfg.sweep_yaml)
    assert cfg.num_jobs > 0
    if cfg.array is not None:
        assert cfg.array > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    cfg = parser.parse_args()
    check_args(cfg)
    main(cfg)
