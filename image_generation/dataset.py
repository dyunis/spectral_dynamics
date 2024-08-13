import os

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

CACHE_DIR = './cache'


def get_dataset(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = MNIST(train=True, device=device, cache_dir=config.cache_dir)
    trainset, valset = split_dataset(dataset)
    return trainset, valset


def split_dataset(dataset, train_frac=0.9):
    # use 10% of training data for validation
    train_set_size = int(len(dataset) * train_frac)
    valid_set_size = len(dataset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    trainset, valset = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)
    return trainset, valset


class MNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, device=torch.device('cpu'), cache_dir=CACHE_DIR):
        cifar_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist = torchvision.datasets.MNIST(root=cache_dir, train=train, download=True, transform=cifar_tfm)
        self.imgs, self.tgts = get_dset_tensor(mnist)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        imgs = self.imgs[ix]
        return imgs, self.tgts[ix]


@torch.no_grad()
def compute_mean_std(dataset):
    all_imgs, all_labels = get_dset_tensor(dataset)
    mean, std = all_imgs.mean(dim=[0, 2, 3]), all_imgs.std(dim=[0, 2, 3])
    return mean, std


@torch.no_grad()
def get_dset_tensor(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, drop_last=False)
    all_imgs = []
    all_labels = []
    for batch in dataloader:
        imgs, labels = batch
        all_imgs.append(imgs)
        all_labels.append(labels)
    all_imgs = torch.cat(all_imgs, axis=0)
    all_labels = torch.cat(all_labels, axis=0)
    return all_imgs, all_labels
