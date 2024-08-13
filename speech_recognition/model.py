import os

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchaudio


class PlModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        d_in = 40  # number of mel features
        num_tokens = 29  # number of tokens in lexicon
        self.model = AsrLSTM(d_in, num_tokens, layers=4, d_hid=256, bidirectional=self.config.bidirectional)
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_ix):
        wavs, texts, wav_lens, text_lens = batch
        out = self.model(wavs)
        log_probs = F.log_softmax(out, dim=-1).transpose(0, 1)
        loss = F.ctc_loss(log_probs, texts, wav_lens, text_lens)
        if loss.isnan().sum() > 0:
            raise ValueError('encountered nans in training')
        return loss

    def validation_step(self, batch, batch_ix):
        wavs, texts, wav_lens, text_lens = batch
        out = self.model(wavs)
        log_probs = F.log_softmax(out, dim=-1).transpose(0, 1)
        loss = F.ctc_loss(log_probs, texts, wav_lens, text_lens)

        # greedy decoding, now (bsz, seq_len)
        toks = log_probs.transpose(0, 1).argmax(dim=-1)

        cer, wer = 0, 0
        num_chars, num_words = 0, 0
        for tok, label in zip(toks, texts):
            tok = tok.unique_consecutive()  # collapse repeats in ctc
            tok = tok[tok != 0]  # ctc blank token is zero
            label = label[label != -1]  # remove padding

            # get chars and words
            pchars = list(self.tokenizer.string(tok.tolist()))
            chars = list(self.tokenizer.string(label.tolist()))
            cer += torchaudio.functional.edit_distance(pchars, chars)
            num_chars += len(chars)

            # split on spaces
            pwords = ''.join(pchars).strip().split()
            words = ''.join(chars).strip().split()
            wer += torchaudio.functional.edit_distance(pwords, words)
            num_words += len(words)

        return {'loss': loss, 'cer': cer, 'wer': wer, 'num_chars': num_chars, 'num_words': num_words}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # validation loss starts increasing around epoch 15 w/o decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.num_epochs)
        ret = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
        return ret

    def inter_layer_pairs(self):
        param_keys = [k for k, p in self.named_parameters() if len(p.shape) >= 2]
        pairs = []
        exts = ['']
        if self.config.bidirectional:
            exts.append('_reverse')
        for layer in range(3):
            for ext1, ext2 in zip(exts, exts):
                # input-hidden interaction
                pairs.append((f'model.lstm.weight_ih_l{layer}{ext1}', f'model.lstm.weight_hh_l{layer}{ext2}'))
                # input-next input interaction
                pairs.append((f'model.lstm.weight_ih_l{layer}{ext1}', f'model.lstm.weight_ih_l{layer+1}{ext2}'))
                # hidden-next input interaction
                pairs.append((f'model.lstm.weight_hh_l{layer}{ext1}', f'model.lstm.weight_ih_l{layer+1}{ext2}'))

        for ext in exts:
            # prev input-output interaction
            pairs.append((f'model.lstm.weight_ih_l3{ext}', 'model.linear.weight'))
            # prev hidden-output interaction
            pairs.append((f'model.lstm.weight_hh_l3{ext}', 'model.linear.weight'))
        return pairs


class AsrLSTM(torch.nn.Module):
    def __init__(self, d_in, d_out, layers=2, d_hid=512, bidirectional=False):
        super(AsrLSTM, self).__init__()
        self.num_dirs = 1 + int(bidirectional)
        self.lstm = torch.nn.LSTM(input_size=d_in, hidden_size=d_hid, num_layers=layers, batch_first=True, bidirectional=bidirectional)
        self.linear = torch.nn.Linear(d_hid * self.num_dirs, d_out)

    def forward(self, x, proj=True):
        x, _ = self.lstm(x)
        if proj:
            x = self.linear(x)
        return x
