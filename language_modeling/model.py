import math
import os

import lightning.pytorch as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class PlModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        # architecture like pythia-160m https://arxiv.org/pdf/2304.01373.pdf
        self.model = TransformerLM(n_tokens=tokenizer.vocab_size, d_model=768, n_heads=12, d_hid=768, n_layers=12, dropout=0.0)

    def training_step(self, batch, batch_ix):
        text, label = batch
        src_mask = generate_square_subsequent_mask(text.shape[1]).to(text.device)
        out = self.model(text, src_mask)
        # out = self.model(input_ids=text).logits
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), label.reshape(-1))
        if loss.isnan().sum() > 0:
            raise ValueError('encountered nans in training')
        return loss

    def validation_step(self, batch, batch_ix):
        text, label = batch
        src_mask = generate_square_subsequent_mask(text.shape[1]).to(text.device)
        out = self.model(text, src_mask)
        # out = self.model(input_ids=text).logits
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)), label.reshape(-1))
        return {'loss': loss}

    def configure_optimizers(self):
        # hyps from https://github.com/EleutherAI/pythia/blob/main/models/160M/pythia-160m.yml
        # smaller beta (20 step tail average vs. 1000 for default) is due to LLMs taking relatively few optimization steps during training
        # unlike larger models (also we probably don't need large tail averaging even in small models, just slows us down)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay, betas=(0.9, 0.95), eps=1e-8)

        if self.config.lr_schedule == 'noam':
            # lr scheduler from Attention is All You Need
            def inv_sqrt_sched(step, warmup_steps=self.config.warmup_steps):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    return warmup_steps ** 0.5 * step ** -0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, inv_sqrt_sched)
        elif self.config.lr_schedule == 'warmup_cosine':
            # schedule from pythia paper
            # https://arxiv.org/pdf/2304.01373.pdf
            # ~222 steps/epoch with 57000 chunks of size 2048 and bsz 256
            total_steps = 30 * 222  # 20 epochs
            def warmup_cosine_decay(step):
                if step < self.config.warmup_steps:
                    return step / self.config.warmup_steps
                elif step > self.config.warmup_steps and step < total_steps:
                    # from 1 to 0.1
                    return 0.45 * (math.cos(math.pi * (step - self.config.warmup_steps) / total_steps) + 1.0) + 0.1
                else:
                    return 0.1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_decay)
        else:
            scheduler = None

        if scheduler is not None:
            ret = {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        else:
            ret = optimizer
        return ret

    # ignores residual connections between layers due to explosion of possibilities
    def inter_layer_pairs(self):
        param_keys = [k for k, p in self.named_parameters() if len(p.shape) >= 2]
        pairs = []
        # input embeds to first attn
        pairs.append(('model.encoder.weight', 'model.transformer_encoder.layers.0.self_attn.in_proj_weight'))
        for i in range(12):
            # W_q to W_k^T and W_qW_k^T to W_v
            pairs.append((f'model.transformer_encoder.layers.{i}.self_attn.in_proj_weight', f'model.transformer_encoder.layers.{i}.self_attn.in_proj_weight'))
            # W_v to W_o
            pairs.append((f'model.transformer_encoder.layers.{i}.self_attn.in_proj_weight', f'model.transformer_encoder.layers.{i}.self_attn.out_proj.weight'))
            # W_o to W_1 of mlp
            pairs.append((f'model.transformer_encoder.layers.{i}.self_attn.out_proj.weight', f'model.transformer_encoder.layers.{i}.linear1.weight'))
            # W_1 to W_2 inside mlp
            pairs.append((f'model.transformer_encoder.layers.{i}.linear1.weight', f'model.transformer_encoder.layers.{i}.linear2.weight'))
            if i < 11:
                # W_2 to W_in of next block
                pairs.append((f'model.transformer_encoder.layers.{i}.linear2.weight', f'model.transformer_encoder.layers.{i+1}.self_attn.in_proj_weight'))
        # last W_2 to output
        pairs.append(('model.transformer_encoder.layers.11.linear2.weight', 'model.decoder.weight'))
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
