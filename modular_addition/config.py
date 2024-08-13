import argparse
import hashlib
import json
import os


def setup_wandb(config):
    import wandb
    if config.wandb_name is None:
        dhash = hashlib.md5()
        encoded = json.dumps(vars(config), sort_keys=True).encode()
        wandb_name = hashlib.md5(encoded).hexdigest()[:8]
    else:
        wandb_name = config.wandb_name

    # specify wandb dir
    if config.wandb_dir is None:
        wandb_dir = 'wandb_folder'
        if 'WANDB_DIR' in os.environ and os.environ['WANDB_DIR'] is not None:
            wandb_dir = os.environ['WANDB_DIR']
    else:
        wandb_dir = os.path.join(config.wandb_dir, config.wandb_project, config.wandb_group)
    os.makedirs(wandb_dir, exist_ok=True)

    # specify individual run savedir
    savedir = os.path.join(wandb_dir, wandb_name)
    os.makedirs(savedir, exist_ok=True)

    # need to generate id independently from name as ids are only allowed once
    # per project, so there will be conflicts if you ever need to delete runs
    if os.path.exists(os.path.join(savedir, 'wandb_id.txt')):
        with open(os.path.join(savedir, 'wandb_id.txt'), 'r') as f:
            wandb_id = f.read()
    else:
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(savedir, 'wandb_id.txt'), 'w') as f:
            f.write(wandb_id)

    # exit if run is finished
    if os.path.exists(os.path.join(savedir, 'done.txt')):
        wandb.finish()
        return

    wandb.init(config=config, project=config.wandb_project, group=config.wandb_group, name=wandb_name, id=wandb_id, resume='allow', dir=wandb_dir, settings=wandb.Settings(start_method='thread'))
    config = wandb.config
    return config, wandb_dir, wandb_name, wandb_id, savedir


parser = argparse.ArgumentParser()
# general args
parser.add_argument('--savedir', type=str, default='test_run', help='savedir for checkpoints')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--time_limit', type=int, default=3.5 * 3600, help='time limit for slurm')
parser.add_argument('--num_gpus', type=int, default=-1, help='specify the number of GPUs, -1 is all available')

# data args
parser.add_argument('--cache_dir', type=str, default='./cache', help='cache directory for data')
parser.add_argument('--data_frac', type=float, default=0.3, help='fraction of data to use')
parser.add_argument('--prime_number', type=int, default=113, help='prime to use for modular addition')

# model args
parser.add_argument('--model_arch', type=str, default='tfmr', choices=['tfmr', 'mlp'], help='which model to use')
parser.add_argument('--remove_layernorm', action='store_true', default=False, help='remove layernorm from transformer model')
parser.add_argument('--ckpt_path', type=str, default=None, help='prior checkpoint to load matching weights from')

# optimization args
parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer to use')
parser.add_argument('--num_epochs', type=int, default=40_000, help='number of epochs to loop through the data')
parser.add_argument('--batch_size', type=int, default=16384, help='want full batch as 113 * 113 = 12769 examples')
parser.add_argument('--lr', type=float, default=1e-3, help='batch size for offline training')
parser.add_argument('--weight_decay', type=float, default=1.0, help='weight decay for optimizer')
parser.add_argument('--lr_schedule', type=str, default=None, choices=[None, 'cosine', 'step', 'warmup'], help='lr schedule to use')
parser.add_argument('--warmup_steps', type=int, default=100, help='warmup steps for cosine schedule')

# logging args
parser.add_argument('--save_interval', type=int, default=100, help='offline epoch interval to save')
parser.add_argument('--eval_interval', type=int, default=100, help='epoch interval to evaluate on dataset')
parser.add_argument('--plot_svs', action='store_true', default=False, help='plot svs during training')

# wandb
parser.add_argument('--use_wandb', action='store_true', default=False, help='log with wandb')
parser.add_argument('--wandb_project', type=str, help='wandb project to log in')
parser.add_argument('--wandb_group', type=str, help='wandb group for runs')
parser.add_argument('--wandb_dir', type=str, help='base wandb directory')
parser.add_argument('--wandb_name', type=str, help='wandb run id')

config = parser.parse_args()
