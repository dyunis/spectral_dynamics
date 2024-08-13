import random

import torch


# from https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/transformers.py
def get_dataset(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_to_generate = config.prime_number
    pairs = [(i, j, num_to_generate, (i+j) % num_to_generate) for i in range(num_to_generate) for j in range(num_to_generate)]
    random.seed(config.seed)
    random.shuffle(pairs)
    div = int(config.data_frac * len(pairs))
    pairs = torch.tensor(pairs).to(device)
    trainset, valset = pairs[:div], pairs[div:]
    return trainset, valset
