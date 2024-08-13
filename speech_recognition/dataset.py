import json
import os

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchvision
from tqdm import tqdm

CACHE_DIR = './cache'


def get_dataset(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 100h clean trainset
    trainset = torchaudio.datasets.LIBRISPEECH(root=config.cache_dir, url='train-clean-100', download=True)
    # clean devset
    valset = torchaudio.datasets.LIBRISPEECH(root=config.cache_dir, url='dev-clean', download=True)

    # simple librispeech character tokenizer
    tokenizer = LibriTokenizer(trainset, config.cache_dir)
    # we use greedy decoding
    decoder = None

    # transforms for input and output
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate = 16000,
        n_fft=400,
        win_length=None,  # 16000 * 25 / 1000 (25ms window)
        hop_length=160,  # 16000 * 10 / 1000 (10ms hop)
        center=True,
        pad_mode='reflect',
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=40,  # 40 filters
        mel_scale='htk'
    )
    transform = torchvision.transforms.Compose([mel_spec, utt_normalize, frame_average])
    text_transform = None

    trainset = TransformedLibriSpeech(trainset, tokenizer=tokenizer, decoder=decoder, transform=transform, text_transform=text_transform, root=config.cache_dir, split='train')
    valset = TransformedLibriSpeech(valset, tokenizer=tokenizer, decoder=decoder, transform=transform, text_transform=text_transform, root=config.cache_dir, split='val')
    return trainset, valset


class TransformedLibriSpeech(torch.utils.data.Dataset):

    @classmethod
    def collate(cls, batch):
        # batch is list of tuples 
        wavs, texts = list(zip(*batch))
        wav_lens = torch.tensor([len(wav) for wav in wavs], dtype=torch.int)
        text_lens = torch.tensor([len(text) for text in texts], dtype=torch.int)
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0.0)
        texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=-1)
        return wavs, texts, wav_lens, text_lens

    def __init__(self, dataset, tokenizer, decoder, transform=None, text_transform=None, root=None, split='train'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.decoder = decoder
        self.transform = transform
        self.text_transform = text_transform
        self.root = root
        assert split in ('train', 'val')
        self.split = split
        self._cached = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.root is not None:
            if not os.path.exists(self.root):
                os.makedirs(self.root)

            # check for transformed features
            if os.path.exists(os.path.join(self.root, 'transformed', self.split, 'done.txt')):
                self._load()
            else:
                self._save()
            self._cached = True

    def _load(self):
        # load transformed feats
        wav_path = os.path.join(self.root, 'transformed', self.split, 'audio')
        text_path = os.path.join(self.root, 'transformed', self.split, 'text')
        self.audio = [
            torch.load(os.path.join(wav_path, f'{ix}.pt'), map_location=self.device)
            for ix in tqdm(range(len(self)), desc='loading audio')
        ]
        self.text = [
            torch.load(os.path.join(text_path, f'{ix}.pt'), map_location=self.device)
            for ix in tqdm(range(len(self)), desc='loading transcripts')
        ]

    def _save(self):
        wav_path = os.path.join(self.root, 'transformed', self.split, 'audio')
        text_path = os.path.join(self.root, 'transformed', self.split, 'text')
        if not os.path.exists(wav_path):
            os.makedirs(wav_path)
        if not os.path.exists(text_path):
            os.makedirs(text_path)

        # can't stack due to different sequence lengths
        self.audio = []
        self.text = []
        for ix in tqdm(range(len(self)), desc='saving transformed data'):
            w, t = self[ix]
            self.audio.append(w.to(self.device))
            self.text.append(t.to(self.device))
            torch.save(w, os.path.join(wav_path, f'{ix}.pt'))
            torch.save(t, os.path.join(text_path, f'{ix}.pt'))

        with open(os.path.join(self.root, 'transformed', self.split, 'done.txt'), 'w') as f:
            f.write('done')

    def __getitem__(self, ix):
        if self._cached:
            waveform = self.audio[ix]
            transcript = self.text[ix]
        else:
            waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[ix]
            if len(waveform.shape) > 1:
                waveform = waveform.mean(dim=0)
            if self.transform is not None:
                waveform = self.transform(waveform)
            if len(waveform.shape) > 1:
                waveform = waveform.transpose(0, 1)  # return [seq_len, hid] tensor

            if self.text_transform is not None:
                transcript = self.text_transform(transcript)
            transcript = self.tokenizer(transcript)
        return waveform, transcript

    def __len__(self):
        return len(self.dataset)


class LibriTokenizer:
    def __init__(self, dataset, savedir):
        self.save_path = os.path.join(savedir, f'librispeech_{len(dataset)}_char2ix.json')
        if os.path.exists(self.save_path):
            self.load()
        else:
            self.char2ix = self.make_char2ix(dataset)
            self.save()

        self.ix2char = {i: c for c, i in self.char2ix.items()}
        self.save()
        self.blank = '<blank>'
        self.unk = '<unk>'
        self.space = ' '
        self.char2ix = {self.blank: 0, self.unk: 1, **{c: i + 2 for c, i in self.char2ix.items()}}
        self.ix2char = {0: self.blank, 1: self.unk, **{i + 2: c for i, c in self.ix2char.items()}}

    def __call__(self, transcript):
        ixs = [self.char2ix[c if c in self.char2ix else self.unk] for c in transcript]
        ixs = torch.tensor(ixs)
        return ixs

    def string(self, ixs):
        chars = [self.ix2char[ix if ix in self.ix2char else 1] for ix in ixs]
        s = ''.join(chars)
        return s

    def __len__(self):
        return len(list(self.char2ix.keys()))

    def make_char2ix(self, dataset):
        chars = set()
        for example in tqdm(dataset, desc='building vocab'):
            _, _, transcript, _, _, _ = example
            chars |= set(transcript)
        chars = sorted(chars)
        char2ix = {c: i for i, c in enumerate(chars)}
        return char2ix

    def save(self):
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
        with open(os.path.join(self.save_path), 'w') as f:
            json.dump(self.char2ix, f, indent=0, sort_keys=True)

    def load(self):
        assert os.path.exists(self.save_path)
        with open(self.save_path, 'r') as f:
            self.char2ix = json.load(f)


def utt_normalize(spec):
    # takes in spectrogram of shape (n_feats, seq_len)
    assert len(spec.shape) == 2
    mean = torch.mean(spec, dim=1, keepdim=True)
    std = torch.std(spec, dim=1, keepdim=True)
    spec = (spec - mean) / std
    return spec


def frame_average(spec, n=3):
    # average consecutive frames in multiples of n, unlike framestacking
    # for reducing time resolution doesn't require triplication of lstm
    # features for all frames stacked
    assert len(spec.shape) == 2
    n_feat, seq_len = spec.shape
    spec = spec[:, :(seq_len // n) * n]
    smoothed = torch.zeros((n_feat, seq_len // n))
    for i in range(seq_len // n):
        smoothed[:, i] = spec[:, i*n:(i+1)*n].mean(dim=-1)
    return smoothed
