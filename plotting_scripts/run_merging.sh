#!/usr/bin/bash
set -e

# merging
python merge_run_data.py ../image_classification/exps/imgclass_vgg_cifar10
python merge_run_data.py ../image_classification/exps/imgclass_vgg_cifar10_wd
python merge_run_data.py ../image_generation/exps/imggen_unet_mnist
python merge_run_data.py ../image_generation/exps/imggen_unet_mnist_wd
python merge_run_data.py ../speech_recognition/exps/speech_lstm_libri
python merge_run_data.py ../speech_recognition/exps/speech_lstm_libri_wd
python merge_run_data.py ../language_modeling/exps/language_tfmr_wikitext
python merge_run_data.py ../language_modeling/exps/language_tfmr_wikitext_wd

# grokking
python merge_run_data.py ../modular_addition/exps/modadd_tfmr_113_nolayernorm
python merge_run_data.py ../modular_addition/exps/modadd_tfmr_113_nolayernorm_wd
python merge_run_data.py ../modular_addition/exps/modadd_tfmr_113_nolayernorm_nowd_frac0.9

# memorization
python merge_run_data.py ../image_classification/exps/imgclass_mlp_cifar10_rethinking

# lottery tickets
python merge_run_data.py ../image_classification/exps/imgclass_vgg_cifar10_mask

# lmc
python merge_lmc_data.py ../image_classification/exps/imgclass_vgg_cifar10_branches/lmc
python merge_lmc_data.py ../image_generation/exps/imggen_unet_mnist_branches/lmc
python merge_lmc_data.py ../speech_recognition/exps/speech_lstm_libri_branches/lmc
python merge_lmc_data.py ../language_modeling/exps/language_tfmr_wikitext_branches/lmc

# lmc cross
python merge_lmc_cross_data.py ../image_classification/exps/imgclass_vgg_cifar10_branches/lmc_cross
python merge_lmc_cross_data.py ../image_generation/exps/imggen_unet_mnist_branches/lmc_cross
python merge_lmc_cross_data.py ../speech_recognition/exps/speech_lstm_libri_branches/lmc_cross
python merge_lmc_cross_data.py ../language_modeling/exps/language_tfmr_wikitext_branches/lmc_cross

# lmc pert
python merge_lmc_pert_data.py ../image_classification/exps/imgclass_vgg_cifar10_branches_pert/lmc
python merge_lmc_pert_data.py ../image_generation/exps/imggen_unet_mnist_branches_pert/lmc
python merge_lmc_pert_data.py ../speech_recognition/exps/speech_lstm_libri_branches_pert/lmc
python merge_lmc_pert_data.py ../language_modeling/exps/language_tfmr_wikitext_branches_pert/lmc
