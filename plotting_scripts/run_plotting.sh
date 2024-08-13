#!/usr/bin/bash
set -e

# plotting
python plot_run.py ../image_classification/exps/imgclass_vgg_cifar10/lr_0.1
python plot_run.py ../image_generation/exps/imggen_unet_mnist/lr_0.0003
python plot_run.py ../speech_recognition/exps/speech_lstm_libri/lr_0.0003
python plot_run.py ../language_modeling/exps/language_tfmr_wikitext/lr_0.0006

# wd plotting
for wd in 0.0 0.001 0.01 0.1 ; do
  python plot_run.py ../image_classification/exps/imgclass_vgg_cifar10_wd/weight_decay_"$wd"
done
for wd in 0.0 0.1 1.0 10.0 ; do
  python plot_run.py ../image_generation/exps/imggen_unet_mnist_wd/weight_decay_"$wd"
  python plot_run.py ../speech_recognition/exps/speech_lstm_libri_wd/weight_decay_"$wd"
  python plot_run.py ../language_modeling/exps/language_tfmr_wikitext_wd/weight_decay_"$wd"
done

# grokking
python plot_run.py ../modular_addition/exps/modadd_tfmr_113_nolayernorm/lr_0.001
python plot_run.py ../modular_addition/exps/modadd_tfmr_113_nolayernorm_wd/weight_decay_0.0
python plot_run.py ../modular_addition/exps/modadd_tfmr_113_nolayernorm_nowd_frac0.9/lr_0.001
python plot_run.py ../modular_addition/exps/modadd_tfmr_97_slingshot/lr_0.001

# memorization
python plot_run.py ../image_classification/exps/imgclass_mlp_cifar10_rethinking/lr_0.001-random_labels_False
python plot_run.py ../image_classification/exps/imgclass_mlp_cifar10_rethinking/lr_0.001-random_labels_True

# lottery tickets
for ms in 0.05 ; do
  for mt in 'magnitude' 'random' ; do
    python plot_run.py ../image_classification/exps/imgclass_vgg_cifar10_mask/mask_sparsity_"$ms"-mask_type_"$mt"
  done
done

# lmc
python plot_lmc.py ../image_classification/exps/imgclass_vgg_cifar10_branches/lmc
python plot_lmc.py ../image_generation/exps/imggen_unet_mnist_branches/lmc
python plot_lmc.py ../speech_recognition/exps/speech_lstm_libri_branches/lmc
python plot_lmc.py ../language_modeling/exps/language_tfmr_wikitext_branches/lmc

# lmc cross
python plot_lmc.py ../image_classification/exps/imgclass_vgg_cifar10_branches/lmc_cross
python plot_lmc.py ../image_generation/exps/imggen_unet_mnist_branches/lmc_cross
python plot_lmc.py ../speech_recognition/exps/speech_lstm_libri_branches/lmc_cross
python plot_lmc.py ../language_modeling/exps/language_tfmr_wikitext_branches/lmc_cross

# lmc pert
python plot_lmc_pert.py ../image_classification/exps/imgclass_vgg_cifar10_branches_pert/lmc
python plot_lmc_pert.py ../image_generation/exps/imggen_unet_mnist_branches_pert/lmc
python plot_lmc_pert.py ../speech_recognition/exps/speech_lstm_libri_branches_pert/lmc
python plot_lmc_pert.py ../language_modeling/exps/language_tfmr_wikitext_branches_pert/lmc
