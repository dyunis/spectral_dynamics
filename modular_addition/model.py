import math
import os

import lightning.pytorch as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class PlModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # architecture from progress measures for grokking: https://arxiv.org/pdf/2301.05217.pdf
        # slight difference in that we use layernorm and they don't
        # 113 tokens for digits, 1 extra for equals sign (to get model to learn commutativity)
        if config.model_arch == 'mlp':
            # this is from https://arxiv.org/pdf/2310.06110.pdf appendix 8.3, the weight decay free experiments
            # but we add more hidden layers to see the concentration effect
            # without, the rank is constrained by input/output dims
            self.model = MLP(d_in=config.prime_number*2, hid=100, d_out=config.prime_number)
        else:
            self.model = TransformerLM(n_tokens=config.prime_number+1, d_model=128, n_heads=4, d_hid=512, n_layers=1)
            if self.config.remove_layernorm:
                replace_modules_with_identity('', self.model, module_type=nn.LayerNorm)  # progress measures paper doesn't use layernorm, with layernorm loss curves are super noisy

    def training_step(self, batch, batch_ix):
        data, label = batch[:, :-1], batch[:, -1:]
        if self.config.model_arch == 'mlp':
            # reshape data to one-hot, then flatten and concatenate
            # matching details from https://arxiv.org/pdf/2310.06110.pdf appendix 8.3, no wd mlp experiments
            data = data[:, :2]
            data = F.one_hot(data, num_classes=self.config.prime_number).flatten(1).float()
            out = self.model(data)
            label = F.one_hot(label, num_classes=self.config.prime_number).flatten(1).float()
            loss = F.mse_loss(out, label)
        else:
            src_mask = generate_square_subsequent_mask(data.shape[1]).to(data.device)
            out = self.model(data, src_mask)
            out = out[:, -1:]  # take last position
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)), label.reshape(-1))
        if loss.isnan().sum() > 0:
            raise ValueError('encountered nans in training')
        return loss

    def validation_step(self, batch, batch_ix):
        data, label = batch[:, :-1], batch[:, -1:]
        if self.config.model_arch == 'mlp':
            data = data[:, :2]
            data = F.one_hot(data, num_classes=self.config.prime_number).flatten(1).float()
            out = self.model(data)
            label = F.one_hot(label, num_classes=self.config.prime_number).flatten(1).float()
            loss = F.mse_loss(out, label)
            label = label.argmax(dim=-1)
        else:
            src_mask = generate_square_subsequent_mask(data.shape[1]).to(data.device)
            out = self.model(data, src_mask)
            out = out[:, -1:]  # take last position
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)), label.reshape(-1))
        err = (out.argmax(dim=-1) != label).float().mean()
        return {'loss': loss, 'err': err}

    def configure_optimizers(self):
        # hyps from
        # https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/transformers.py
        # smaller beta (50 step tail average vs. 1000 for default) is due to LLMs taking relatively few optimization steps during training
        # unlike larger models (also we probably don't need large tail averaging even in small models, just slows us down)
        if self.config.optim == 'sgd':
            assert self.config.model_arch == 'mlp'
            assert self.config.prime_number == 23
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay, betas=(0.9, 0.98), eps=1e-8)

        # schedule not mentioned in paper, but from https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/transformers.py
        #
        # use cosine decay instead for less unstable loss at end of training
        scheduler = None
        if self.config.lr_schedule == 'cosine':
            total_steps = self.config.num_epochs
            def warmup_cosine_decay(step):
                if step < self.config.warmup_steps:
                    return step / self.config.warmup_steps
                elif step > self.config.warmup_steps and step < total_steps:
                    # from 1 to 0.1
                    return 0.45 * (math.cos(math.pi * (step - self.config.warmup_steps) / total_steps) + 1.0) + 0.1
                else:
                    return 0.1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_decay)
            interval = 'epoch'
        elif self.config.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10_000, 20_000, 30_000], gamma=0.1)
            interval = 'epoch'
        elif self.config.lr_schedule == 'warmup':
            interval = 'step'
            # linear warmup for first 10 steps in slingshot paper appendix
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))

        if scheduler is not None:
            ret = {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': interval,
                    'frequency': 1,
                }
            }
        else:
            ret = optimizer
        return ret

    # ignores residual connections between layers due to explosion of possibilities
    def inter_layer_pairs(self):
        if self.config.model_arch == 'mlp':
            param_keys = [k for k, p in self.named_parameters() if len(p.shape) >= 2]
            pairs = [(param_keys[i], param_keys[i+1]) for i in range(len(param_keys)-1)]
        else:
            param_keys = [k for k, p in self.named_parameters() if len(p.shape) >= 2]
            pairs = []
            # input embeds to first attn
            pairs.append(('model.encoder.weight', 'model.transformer_encoder.layers.0.self_attn.in_proj_weight'))
            for i in range(1):
                # W_q to W_k^T and W_qW_k^T to W_v
                pairs.append((f'model.transformer_encoder.layers.{i}.self_attn.in_proj_weight', f'model.transformer_encoder.layers.{i}.self_attn.in_proj_weight'))
                # W_v to W_o
                pairs.append((f'model.transformer_encoder.layers.{i}.self_attn.in_proj_weight', f'model.transformer_encoder.layers.{i}.self_attn.out_proj.weight'))
                # W_o to W_1 of mlp
                pairs.append((f'model.transformer_encoder.layers.{i}.self_attn.out_proj.weight', f'model.transformer_encoder.layers.{i}.linear1.weight'))
                # W_1 to W_2 inside mlp
                pairs.append((f'model.transformer_encoder.layers.{i}.linear1.weight', f'model.transformer_encoder.layers.{i}.linear2.weight'))
                if i < 0:
                    # W_2 to W_in of next block
                    pairs.append((f'model.transformer_encoder.layers.{i}.linear2.weight', f'model.transformer_encoder.layers.{i+1}.self_attn.in_proj_weight'))
            # last W_2 to output
            pairs.append(('model.transformer_encoder.layers.0.linear2.weight', 'model.decoder.weight'))
        return pairs


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TransformerLM(nn.Module):

    def __init__(self, n_tokens: int, d_model: int, n_heads: int, d_hid: int,
                 n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.Embedding(n_tokens, d_model)
        # use pre-ln because it's more stable
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, n_heads, d_hid, dropout, norm_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_model, n_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        # I assume [batch_size, seq_len] for src
        src = src.transpose(0, 1)
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, n_tokens]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        # transpose to [batch_size, seq_len, n_tokens]
        output = output.transpose(0, 1)
        return output

    @torch.no_grad()
    def generate(self, src=None, seq_len=32):
        self.eval()
        device = next(self.parameters()).device
        if src is not None:
            assert len(src.shape) == 2
        else:
            generator = torch.Generator()
            generator.manual_seed(0)

            # select random start token index
            src = torch.randint(self.encoder.shape[0], generator=generator)
            src.unsqueeze(0)  # bsz 1
        src.to(device)

        src_len = src.shape[1]
        for i in range(seq_len - src_len):
            src_mask = generate_square_subsequent_mask(src.shape[1]).to(device)
            out = self.forward(src, src_mask)
            next_ix = torch.argmax(out[:, -1], dim=-1).unsqueeze(1)
            src = torch.cat([src, next_ix], dim=1)
        return src


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MLP(torch.nn.Module):
    def __init__(self, d_in=23*2, hid=256, d_out=23):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(d_in, hid),
            torch.nn.Linear(hid, hid),
            torch.nn.Linear(hid, hid),
            torch.nn.Linear(hid, hid),
            torch.nn.Linear(hid, d_out),
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x)
            else:
                x = torch.nn.functional.relu(layer(x))
        return x


@torch.no_grad()
def replace_modules_with_identity(name, module, module_type=None):
    mods = [name for name, mod in module._modules.items() if isinstance(mod, module_type)]
    for name in mods:
        target_attr = module._modules[name]
        module._modules[name] = nn.Identity()
    # this step needed to recursively explore module lists
    for name, mod in module._modules.items():
        replace_modules_with_identity(name, mod, module_type=module_type)
