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

        if self.config.model_arch == 'unet':
            self.model = UNet_res()
        else:
            self.model = MLPConditionalModel(self.config.timesteps)

        beta_min = 1e-5
        beta_max = 0.5e-2
        betas = torch.linspace(-6, 6, config.timesteps)
        self.register_buffer('betas', torch.sigmoid(betas) * (beta_max - beta_min) + beta_min)
        self.register_buffer('alphas', 1 - self.betas)
        alphas_prod = torch.cumprod(self.alphas, 0)
        self.register_buffer('alphas_bar_sqrt', torch.sqrt(alphas_prod))
        self.register_buffer('one_minus_alphas_bar_sqrt', torch.sqrt(1 - alphas_prod))

        self.ema = None
        if config.ema_gamma > 0.0:
            self.ema = EMA(config.ema_gamma)
            self.ema.register(self.model)
            self.ema.to(torch.device('cuda:0'))

    def training_step(self, batch, batch_ix):
        # update ema
        if self.ema is not None:
            self.ema.update(self.model)

        # training step
        imgs, _ = batch
        t = torch.randint(0, self.config.timesteps, size=(imgs.size(0),), device=imgs.device)
        a = self.extract(self.alphas_bar_sqrt, t, imgs)
        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, imgs)
        e = torch.randn_like(imgs)
        noised = imgs * a + e * am1
        denoised = self.model(noised, t)

        loss = (e - denoised).square().mean()
        if loss.isnan().sum() > 0:
            raise ValueError('encountered nans in training')
        return loss

    def validation_step(self, batch, batch_ix):
        imgs, _ = batch
        t = torch.randint(0, self.config.timesteps, size=(imgs.size(0),), device=imgs.device)
        a = self.extract(self.alphas_bar_sqrt, t, imgs)
        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, imgs)
        e = torch.randn_like(imgs)
        noised = imgs * a + e * am1
        denoised = self.model(noised, t)

        loss = (e - denoised).square().mean()
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optimizer

    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    # one reverse diffusion step
    def p_sample(self, x, t, deterministic=False):
        t = torch.tensor([t]).to(x.device)
        # Factor to the model output
        eps_factor = ((1 - self.extract(self.alphas, t, x)) / self.extract(self.one_minus_alphas_bar_sqrt, t, x))
        # Model output
        eps_theta = self.model(x, t)
        # Final values
        mean = (1 / self.extract(self.alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
        # Generate z
        z = torch.randn_like(x)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, x).sqrt()
        if deterministic:
            sigma_t = 0
        sample = mean + sigma_t * z
        return (sample)

    # whole reverse diffusion chain
    def p_sample_loop(self, shape, cur_x=None):
        if cur_x is None:
            cur_x = torch.randn(shape, device=self.betas.device)
        x_seq = [cur_x]
        for i in reversed(range(self.config.timesteps)):
            cur_x = self.p_sample(cur_x, i)
            x_seq.append(cur_x)
        return x_seq

    # ignores residual connections between layers?
    def inter_layer_pairs(self):
        param_keys = [k for k, p in self.named_parameters() if len(p.shape) >= 2]
        pairs = []

        # downsampling path
        pairs.append(('model.conv1.weight', 'model.conv2.weight'))
        pairs.append(('model.conv2.weight', 'model.conv3.weight'))
        pairs.append(('model.conv3.weight', 'model.conv4.weight'))

        # downsampling conditioning path
        pairs.append(('model.dense1.dense.weight', 'model.conv2.weight'))
        pairs.append(('model.dense2.dense.weight', 'model.conv3.weight'))
        pairs.append(('model.dense3.dense.weight', 'model.conv4.weight'))
        pairs.append(('model.dense4.dense.weight', 'model.tconv4.weight'))

        # middle
        pairs.append(('model.conv4.weight', 'model.tconv4.weight'))

        # upsampling path
        pairs.append(('model.tconv4.weight', 'model.tconv3.weight'))
        pairs.append(('model.tconv3.weight', 'model.tconv2.weight'))
        pairs.append(('model.tconv2.weight', 'model.tconv1.weight'))

        # skip connections
        pairs.append(('model.conv1.weight', 'model.tconv1.weight'))
        pairs.append(('model.dense1.dense.weight', 'model.tconv1.weight'))
        pairs.append(('model.conv2.weight', 'model.tconv2.weight'))
        pairs.append(('model.dense2.dense.weight', 'model.tconv2.weight'))
        pairs.append(('model.conv3.weight', 'model.tconv3.weight'))
        pairs.append(('model.dense3.dense.weight', 'model.tconv3.weight'))
        return pairs


# modified from https://github.com/azad-academy/denoising-diffusion-model/blob/main/model.py
class MLPConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(MLPConditionalModel, self).__init__()
        self.lin1 = ConditionalLinear(1 * 28 * 28, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, 1 * 28 * 28)

    def forward(self, x, y):
        x = x.flatten(1)
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        x = self.lin4(x).reshape(x.size(0), 1, 28, 28)
        return x


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out


# from https://colab.research.google.com/drive/1Y5wr91g5jmpCDiX-RLfWL1eSBWoSuLqO?usp=sharing#scrollTo=9is-DXZYwIIi
import functools

import numpy as np
import torch
from torch import nn

# Alternative time-dependent score-based model (double click to expand or collapse)
# uses residual connections
class UNet_res(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, channels=[64, 128, 256, 512], embed_dim=512):
    """Initialize a time-dependent score-based network.

    Args:
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()

    # Gaussian random feature embedding layer for time
    self.time_embed = nn.Sequential(
          GaussianFourierProjection(embed_dim=embed_dim),
          nn.Linear(embed_dim, embed_dim)
          )
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)     #  + channels[2]
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)     #  + channels[1]
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1) #  + channels[0]

    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)

  def forward(self, x, t, y=None):
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.time_embed(t))    
    # Encoding path
    h1 = self.conv1(x)  + self.dense1(embed)   
    ## Incorporate information from t
    ## Group normalization
    h1 = self.act(self.gnorm1(h1))
    h2 = self.conv2(h1) + self.dense2(embed)
    h2 = self.act(self.gnorm2(h2))
    h3 = self.conv3(h2) + self.dense3(embed)
    h3 = self.act(self.gnorm3(h3))
    h4 = self.conv4(h3) + self.dense4(embed)
    h4 = self.act(self.gnorm4(h4))

    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.act(self.tgnorm4(h))
    h = self.tconv3(h + h3)
    h += self.dense6(embed)
    h = self.act(self.tgnorm3(h))
    h = self.tconv2(h + h2)
    h += self.dense7(embed)
    h = self.act(self.tgnorm2(h))
    h = self.tconv1(h + h1)
    return h


class UNet_res_small(nn.Module):
  def __init__(self, c_in=3, channels=[64, 128], embed_dim=128, width=32):
    super().__init__()

    # Gaussian random feature embedding layer for time
    self.embed = nn.Linear(width * width * c_in, embed_dim)

    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(c_in, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

    # Decoding layers where the resolution increases
    self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)     #  + channels[1]
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0], c_in, 3, stride=1) #  + channels[0]

    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)

  def forward(self, x, t, y=None): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t.flatten(1)))
    # Encoding path
    h1 = self.conv1(x)  + self.dense1(embed)   
    ## Incorporate information from t
    ## Group normalization
    h1 = self.act(self.gnorm1(h1))
    h2 = self.conv2(h1) + self.dense2(embed)
    h2 = self.act(self.gnorm2(h2))

    # Decoding path
    h = self.tconv2(h2)
    ## Skip connection from the encoding path
    h += self.dense7(embed)
    h = self.act(self.tgnorm2(h))
    h = self.tconv1(h + h1)
    return h


class UNet_res_small_conv(nn.Module):
  def __init__(self, c_in=3, channels=[64, 128], embed_dim=128, width=32):
    super().__init__()

    # Gaussian random feature embedding layer for time
    # self.embed = nn.Linear(width * width * c_in, embed_dim)
    self.embed = nn.Conv2d(c_in, embed_dim, 3, 1, 1)

    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(c_in, channels[0], 3, stride=1, bias=False)
    # self.dense1 = Dense(embed_dim, channels[0])
    self.dense1 = nn.Conv2d(embed_dim, channels[0], kernel_size=3, stride=1)
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    # self.dense2 = Dense(embed_dim, channels[1])
    self.dense2 = nn.Conv2d(channels[0], channels[1], 3, stride=2)
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

    # Decoding layers where the resolution increases
    self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)     #  + channels[1]
    # self.dense7 = Dense(embed_dim, channels[0])
    self.dense7 = nn.Conv2d(embed_dim, channels[0], 3, stride=1)
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0], c_in, 3, stride=1) #  + channels[0]

    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)

  def forward(self, x, t, y=None): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))
    # Encoding path
    e1 = self.dense1(embed)
    h1 = self.conv1(x)  + e1
    ## Incorporate information from t
    ## Group normalization
    h1 = self.act(self.gnorm1(h1))
    e2 = self.dense2(self.dense1(embed))
    h2 = self.conv2(h1) + e2
    h2 = self.act(self.gnorm2(h2))

    # Decoding path
    h = self.tconv2(h2)
    ## Skip connection from the encoding path
    h += self.dense7(embed)
    h = self.act(self.tgnorm2(h))
    h = self.tconv1(h + h1)
    return h


#@title Get some modules to let time interact
class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights (frequencies) during initialization.
    # These weights (frequencies) are fixed during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    # Cosine(2 pi freq x), Sine(2 pi freq x)
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps.
  Allow time repr to input additively from the side of a convolution layer.
  """
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]
    # this broadcast the 2d tensor to 4d, add the same value across space.


class EMA(nn.Module):
    def __init__(self, mu=0.999):
        super().__init__()
        self.mu = mu

    def register(self, module):
        # register as buffers so that they're saved by state_dict
        for name, param in module.named_parameters():
            if param.requires_grad:
                buffer_name = name.replace('.', '-')  # buffer names can't contain dots
                self.register_buffer(buffer_name, param.data.clone())

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                buffer_name = name.replace('.', '-')
                getattr(self, buffer_name).copy_(
                    (1. - self.mu) * param.data + self.mu * getattr(self, buffer_name).data
                )

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                buffer_name = name.replace('.', '-')
                param.data.copy_(getattr(self, buffer_name))
