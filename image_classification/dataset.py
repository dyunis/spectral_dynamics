import os

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

CACHE_DIR = './cache'


def get_dataset(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if config.dataset == 'mnist':
        dataset = MNIST(train=True, device=device, cache_dir=config.cache_dir)
        trainset, valset = split_dataset(dataset)
    elif config.dataset == 'cifar10':
        dataset = CIFAR10(train=True, device=device, random_labels=config.random_labels, label_noise=config.label_noise, cache_dir=config.cache_dir)
        trainset, valset = split_dataset(dataset)
    elif config.dataset == 'cifar100':
        dataset = CIFAR100(train=True, device=device, cache_dir=config.cache_dir)
        trainset, valset = split_dataset(dataset)
    else:
        raise NotImplementedError()
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


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, train=True, device=torch.device('cpu'), random_labels=False, label_noise=-1, cache_dir=CACHE_DIR):
        cifar_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        cifar = torchvision.datasets.CIFAR10(root=cache_dir, train=train, download=True, transform=cifar_tfm)
        self.imgs, self.tgts = get_dset_tensor(cifar)
        self.random_labels = random_labels
        if random_labels and label_noise == -1:
            generator = torch.Generator().manual_seed(12345)
            ixs = torch.randperm(len(self.tgts), generator=generator)
            self.tgts = self.tgts[ixs]
        elif label_noise != -1:
            assert 0.0 <= label_noise and label_noise <= 1.0
            # randomize p fraction of labels
            generator = torch.Generator().manual_seed(12345)
            num_random = int(len(self.tgts) * label_noise)
            rand_ixs = torch.randperm(len(self.tgts), generator=generator)[:num_random]
            rand_labels = torch.randint(low=0, high=10, size=(num_random,), generator=generator)
            self.tgts[rand_ixs] = rand_labels
        self.imgs, self.tgts = self.imgs.to(device), self.tgts.to(device)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ix):
        imgs = self.imgs[ix]
        return imgs, self.tgts[ix]


class CIFAR100(torch.utils.data.Dataset):
    def __init__(self, train=True, device=torch.device('cpu'), cache_dir=CACHE_DIR):
        cifar_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        cifar = torchvision.datasets.CIFAR100(root=cache_dir, train=train, download=True, transform=cifar_tfm)
        self.imgs, self.tgts = get_dset_tensor(cifar)

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


if __name__ == '__main__':
    # dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
    # mnist normalize: (0.1307,), (0.3081,)
    # cifar10 normalize: (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    # cifar100 normalize: (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)

    # get items from dataset, log to see class choices are correct
    # they are
    from PIL import Image
    import numpy as np
    arr = dset[0][0].numpy()
    arr = arr.transpose(1, 2, 0)
    arr = (arr * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save('cls0.png')

    arr = dset[len(dset)-1][0].numpy()
    arr = arr.transpose(1, 2, 0)
    arr = (arr * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save('cls1.png')
