from glob import glob
import math
import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class PlModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.dataset == 'cifar100':
            d_out = 100
        else:
            d_out = 10

        if config.model_arch == 'resnet':
            self.model = resnet_from_name('cifar_resnet_20', outputs=d_out)
        elif config.model_arch == 'lenet':
            if config.dataset == 'mnist':
                c_in = 1
            else:
                c_in = 3
            self.model = LeNet(c_in, d_out)
        if config.model_arch == 'vgg':
            self.model = vgg_from_name('cifar_vgg_16', outputs=d_out, batch_norm=True)
        else:
            if config.dataset == 'mnist':
                d_in = 28 * 28
            else:
                d_in = 32 * 32 * 3
            self.model = MLP(layers=4, d_in=d_in, hid=256, d_out=d_out)

        self.augs = None
        if config.he_augs:
            # originally https://arxiv.org/pdf/1512.03385.pdf section 4.2
            self.augs = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(32, 4),
            ])

    def training_step(self, batch, batch_ix):
        imgs, labels = batch
        if self.augs is not None:
            imgs = self.augs(imgs)
        if self.config.model_arch == 'mlp':
            imgs = imgs.view(imgs.size(0), -1)
        out = self.model(imgs)
        loss = torch.nn.functional.cross_entropy(out, labels)
        if loss.isnan().sum() > 0:
            raise ValueError('encountered nans in training')
        return loss

    def validation_step(self, batch, batch_ix):
        imgs, labels = batch
        if self.config.model_arch == 'mlp':
            imgs = imgs.view(imgs.size(0), -1)
        out = self.model(imgs)
        loss = torch.nn.functional.cross_entropy(out, labels)
        preds = out.argmax(dim=-1)
        err = (preds != labels).float().mean()
        return {'loss': loss, 'err': err}

    def configure_optimizers(self):
        if self.config.optim == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay, momentum=self.config.momentum)

        if self.config.lr_schedule == 'step':
            # decay lr by 10x at these iterations (from jonathan's paper)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [82, 123, 160], gamma=0.1)
        elif self.config.lr_schedule == 'decay':
            # from https://openreview.net/pdf?id=Sy8gdB9xx appendix A
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
        else:
            scheduler = None

        if scheduler is not None:
            ret = {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        else:
            ret = optimizer
        return ret


class MLP(torch.nn.Module):
    def __init__(self, layers=3, d_in=1, hid=256, d_out=1):
        assert layers >= 0
        super(MLP, self).__init__()
        decoder_in = d_in

        if layers == 1:
            self.layers = [torch.nn.Linear(d_in, d_out)]
        elif layers == 2:
            self.layers = [torch.nn.Linear(d_in, hid), torch.nn.Linear(hid, d_out)] 
        else:
            self.layers = []
            self.layers.append(torch.nn.Linear(d_in, hid))
            for i in range(layers - 2):
                self.layers.append(torch.nn.Linear(hid, hid))
            self.layers.append(torch.nn.Linear(hid, d_out))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x, proj=True):
        if len(x.shape) > 2:
            x = x.flatten(1)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1: x = layer(x)
            else: x = torch.nn.functional.relu(layer(x))
        return x


# from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class LeNet(torch.nn.Module):
    def __init__(self, c_in=3, c_out=10, dropout=0.0):
        super(LeNet, self).__init__()
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(c_in, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, c_out)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        if self.use_dropout:
            x = self.dropout(x)
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc3(x)
        return x


# code below modified from jonathan frankle's code base https://github.com/facebookresearch/open_lth/blob/master/models/cifar_resnet.py
class ResNet(torch.nn.Module):
    def __init__(self, plan, initializer=None, outputs=None):
        super(ResNet, self).__init__()
        out_dim = outputs or 10
        
        current_filters = plan[0][0]
        self.conv = torch.nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(current_filters)

        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(ResNetBlock(current_filters, filters, downsample))
                current_filters = filters
        
        self.blocks = torch.nn.Sequential(*blocks)
        
        self.fc = torch.nn.Linear(plan[-1][0], out_dim)
        if initializer is not None:
            self.apply(initializer)
    
    def forward(self, x):
        out = torch.nn.functional.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = torch.nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, downsample=False):
        super(ResNetBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_dim)
        self.conv2 = torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_dim)

        if downsample or in_dim != out_dim:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, bias=False),
                torch.nn.BatchNorm2d(out_dim)
            )
        else:
            self.shortcut = torch.nn.Sequential()
    
    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.nn.functional.relu(out)

def valid_model_name(model_name):
    return (model_name.startswith('cifar_resnet_') and
            5 > len(model_name.split('_')) > 2 and
            all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:5]]) and
            (int(model_name.split('_')[2]) - 2) % 6 == 0 and
            int(model_name.split('_')[2]) > 2)

def resnet_from_name(model_name, initializer=None, outputs=10):
    if not valid_model_name(model_name):
        raise ValueError(f'Invalid model name: {model_name}')
    
    name = model_name.split('_')
    W = 16 if len(name) == 3 else int(name[3])
    D = int(name[2])
    if (D - 2) % 3 != 0:
        raise ValueError(f'Invalid ResNet depth: {D}')
    D = (D - 2) // 6
    plan = [(W, D), (2 * W, D), (4 * W, D)]
    return ResNet(plan, initializer=initializer, outputs=outputs)

def conv2d_input_shapes(module, x, key=None):
    input_shapes = {}
    handles = {}
    def hook_fn(m, i, o):
        m._input_shape = i[0].shape[1:] # C_in, H, W (ignore batch dimension)
    with torch.no_grad():
        conv2ds = list(filter(lambda tup: isinstance(tup[1], torch.nn.Conv2d), module.named_modules()))
        for name, conv2d in conv2ds:
            handles[name] = conv2d.register_forward_hook(hook_fn)
        _ = module(x)
        for name, conv2d in conv2ds:
            input_shapes[name] = tuple(conv2d._input_shape)
            delattr(conv2d, '_input_shape')
            handles[name].remove()
    return input_shapes


# code modified from jonathan frankle's https://github.com/facebookresearch/open_lth/blob/master/models/cifar_vgg.py
class Model(torch.nn.Module):
    """A VGG-style neural network designed for CIFAR-10."""

    class ConvModule(torch.nn.Module):
        """A single convolutional module in a VGG network."""

        def __init__(self, in_filters, out_filters, batch_norm=False):
            super(Model.ConvModule, self).__init__()
            self.conv = torch.nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            self.batch_norm = batch_norm
            if batch_norm:
                self.bn = torch.nn.BatchNorm2d(out_filters)

        def forward(self, x):
            out = self.conv(x)
            if self.batch_norm:
                out = self.bn(out)
            return torch.nn.functional.relu(out)

    def __init__(self, plan, initializer=None, outputs=10, batch_norm=False):
        super(Model, self).__init__()

        layers = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Model.ConvModule(filters, spec, batch_norm=batch_norm))
                filters = spec

        self.layers = torch.nn.Sequential(*layers)
        self.fc = torch.nn.Linear(512, outputs)
        self.criterion = torch.nn.CrossEntropyLoss()

        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        x = self.layers(x)
        x = torch.nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_vgg_') and
                len(model_name.split('_')) == 3 and
                model_name.split('_')[2].isdigit() and
                int(model_name.split('_')[2]) in [11, 13, 16, 19])

def vgg_from_name(model_name, initializer=None, outputs=10, batch_norm=False):
    if not Model.is_valid_model_name(model_name):
        raise ValueError('Invalid model name: {}'.format(model_name))

    outputs = outputs or 10

    num = int(model_name.split('_')[2])
    if num == 11:
        plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 13:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 16:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    elif num == 19:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    else:
        raise ValueError('Unknown VGG model: {}'.format(model_name))

    return Model(plan, initializer=initializer, outputs=outputs, batch_norm=batch_norm)
