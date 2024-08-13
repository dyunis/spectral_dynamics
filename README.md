## File list

We make no guarantee that the code will run out of the box as the
merging/plotting depends on the yaml launcher, which is specific to a SLURM
setup. Still, it is straightforward to train individual models, and all the
plotting code can be abstracted.

The code is quite redundant and structured in the following fashion:
- `README.md` - this file
- `[task]/`
  - `config.py` - commandline arguments and logging setup
  - `run.py` - main training script
  - `model.py` - code related to model and optimizer
  - `dataset.py` - code related to data
  - `callbacks.py` - pytorch lightning callbacks needed for saving/logging/misc
  - `evaluate_run.py` - main evaluation script, including SV calculation and layer alignment
  - `evaluate_lmc.py` - main lmc evaluation script
  - `evaluate_lmc_pert.py` - lmc evaluation for perturbed runs
  - `make_branches_yaml.py` - script to generate yaml for branches for lmc experiments
  - `make_eval_yaml.py` - script to generate yaml for eval job
  - `make_eval_lmc_yaml.py` - script to generate yaml for lmc job
  - `make_eval_lmc_cross_yaml.py` - script to generate yaml for lmc between modes job
  - `make_eval_lmc_pert_yaml.py` - script to generate yaml for perturbed lmc job
  - `slurm_run.py` - script to launch an entire yaml of jobs on slurm
  - `sweeps/` - different yaml files that specify hyperparameters of training jobs for experiments in the paper
- `plotting_scripts/`
  - `merge_run_data.py` - script to merge eval data across seeds/within hyperparameter combos
  - `merge_lmc_data.py` - script to merge lmc data across seeds/within splits
  - `merge_lmc_cross_data.py` - script to merge lmc data across modes
  - `merge_lmc_pert_data.py` - script to merge perturbed lmc data
  - `plot_single_run.py` - script to plot a single directory run
  - `plot_run.py` - plotting script for a directory of runs that has been merged
  - `plot_lmc.py` - plotting script for lmc jobs
  - `plot_lmc_pert.py` - plotting script for perturbed lmc jobs
  - `run_merging.sh` - merging script for all jobs to be run
  - `run_plotting.sh` - plotting script for all jobs to be run
where `[task]` is one of `image_classification`, `image_generation`,
`speech_recognition`, `language_modeling` or `modular_addition`.

There are fewer scripts in `modular_addition` as there are no LMC jobs for this
task. Also there is additional code in `image_classification` corresponding to
the lottery ticket jobs:
- `image_classification/`
  - `...`
  - `evaluate_run_mask.py` - lottery ticket eval script
  - `make_eval_mask_yaml.py` - script to generate yaml for lottery ticket eval

## Installation

The following options are provided for installing the packages needed for
running the code. We use `Python 3.11.5`, though the Apptainer installation
should give the most reproducibility. If you are not familiar, Apptainer is
similar to Docker but preserves the original user permissions.

### conda

```bash
# install miniconda
conda install -c pytorch -c nvidia pytorch pytorch-cuda=11.8 torchvision torchaudio

pip install lightning==2.0.6  # for trainer
pip install datasets  # for wikitext-103 dataset
pip install matplotlib  # for visualization
pip install wandb  # for experiment tracking
pip install seaborn  # for pseudo-perceptually uniform colormaps
```

### Apptainer

First make sure you have Apptainer installed. Then follow the instructions
below.

```bash
mkdir ./containers  # we will store installed containers here
cd ./containers

apptainer build --sandbox spectral_dynamics_container docker://nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
# sudo is needed for --writable, but makes it impossible to remove the directory later
sudo apptainer shell --no-home --nv --writable spectral_dynamics_container  # --nv binds gpu access within container, if home dir is nfs --no-home circumvents issues

# below is inside container
apt update
apt upgrade
apt install wget
apt install vim

# need this for installing conda inside the container and making it accessible
mkdir /conda_tmp
cd /conda_tmp

# from slurm docs
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /conda_tmp/mc3
rm Miniconda3-latest-Linux-x86_64.sh
eval "$(/conda_tmp/mc3/bin/conda 'shell.bash' 'hook')"

conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 torchvision torchaudio

pip install lightning==2.0.6  # for trainer, slurm with multi-gpu is weird with later versions
pip install datasets  # for wikitext103 dataset
pip install matplotlib  # for visualization
pip install wandb  # for experiment tracking
pip install seaborn  # for pseudo-perceptually uniform colormaps

exit

# below is outside container
sudo apptainer build spectral_dynamics_container.sif spectral_dynamics_container/
cd ..
```

## Running the code

All experiments were run on an internal SLURM cluster, so it will be most
convenient if you have access, but the code can still be run on a single
machine. In addition the experiments save every intermediate checkpoint, thus
the total storage required for all experiments is on the order of 10T, so if
you are low on hard drive space it may be difficult to replicate all runs.

If using conda, source your installed conda environment. If using apptainer,
run the following to source the environment:
```bash
# need --nv for gpu 
# mount instructions: https://apptainer.org/docs/user/main/bind_paths_and_mounts.html
apptainer shell --nv ./containers/spectral_dynamics_container.sif
source /conda_tmp/mc3/bin/activate
```

### Training

For illustration purposes, we'll use the code in `image_classification/` but
the other task code will work similarly.

To train an individual job:
```bash
cd image_classification
python run.py
```

For a full list of arguments and the defaults see `config.py` in the same
directory. This will generate a directory (use `--savedir` to specify,
`test_run/` is the default) which will save the config used and intermediate
checkpoints generated during training.

To launch a grid search sweep on a SLURM cluster, try:
```bash
python slurm_run.py sweeps/imgclass_vgg_cifar10.yaml
```
see `slurm_run.py` for a full list of arguments allowed including partition,
though all jobs should only require a single GPU with less than 12g VRAM.

This file will generate a directories like
`./exps/imgclass_vgg_cifar10/imgclass_vgg_cifar10-[0, 1, 2, ...]` which will
correspond to the different hyperparameters specified in the grid search sweep yaml.
These directories contain the respective config and checkpoints, and additionally
will contain a `slurm/` directory with the corresponding slurm scripts and
logs for that job. In particular `slurm/sbatch_script.sh` can be relaunched
manually as:
```bash
sbatch sbatch_script.sh
```

### Evaluating

To evaluate a single run that was saved in `test_run/`, try
```bash
python evaluate_run.py -d test_run/
```
which will run all the evaluations and save a file at `test_run/data.npz`.

In order to use SLURM to evaluate all runs in parallel, first we need to create
the eval yaml, which can be done dynamically as:
```bash
python make_eval_yaml.py sweeps/imgclass_vgg_cifar10.yaml
```

and then we can launch the evals as
```bash
python slurm_run.py sweeps/imgclass_vgg_cifar10_eval.yaml
```

### Merging

Running multiple trials for different hyperparameters can get complicated to
manage, especially with LMC jobs, which is why there are merging scripts to
collect data from evaluations with the same hyperparameter combo but different
seeds. To use these scripts on an already evaluated job, try
```
cd ../plotting_scripts
python merge_run_data.py ../image_classification/exps/imgclass_vgg_cifar10
```
which will create directories and files for different hyperparameter combos as
`image_classification/exps/imgclass_vgg_cifar10/[argname]_[argval](-[argname2]_[argval2]...)/[mean.npz, std.npz]`.
When a sweep only varies the random seed, the directory will be the learning rate:
`image_classification/exps/imgclass_vgg_cifar10/lr_0.1/`.

You can see the full list of merging run in `plotting_scripts/run_merging.sh`,
which will differ slightly for LMC jobs.

### Plotting

To plot a single run, try:
```bash
python plot_single_run.py ../image_classification/exps/imgclass_vgg_cifar10/lr_0.1
```

To plot multiple seeds of a merged task directory, try:
```bash
python plot_run.py ../image_classification/exps/imgclass_vgg_cifar10/lr_0.1
```
which will generate the plots in a directory `plotting_scripts/plots/imgclass_vgg_cifar10-lr_0.1`
with the structure:
- `[train, val]-[loss, err].png` - training/validation loss/error along with low-rank pruning evals
- `sv/` - singular value evolution plots and effective rank over layers plot
- `sva/` - singular value agreement plots and stability measure over layers plot
- `align/` - consecutive layer alignment plots and alignment over layers plot
- `metrics/` - effective rank plots for different layers

You can see the full list of plotting calls made in `plotting_scripts/run_plotting.sh`,
which again will differ for LMC jobs slightly.
