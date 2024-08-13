import json
import os

from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from tqdm import tqdm

CACHE_DIR = './cache'


def get_dataset(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', cache_dir=config.cache_dir)
    trainset = LMDataset(tokenizer, dataset_name=config.dataset, chunk_size=config.chunk_size, split='train', cache_dir=config.cache_dir, device=device)
    valset = LMDataset(tokenizer, dataset_name=config.dataset, chunk_size=config.chunk_size, split='validation', cache_dir=config.cache_dir, device=device)
    return trainset, valset


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dataset_name='wikitext2', chunk_size=512, split='train', cache_dir=CACHE_DIR, device=torch.device('cpu')):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size
        self.split = split
        self.cache_dir = cache_dir
        self.device = device

        self.save_path = os.path.join(cache_dir, f'gptneox_toks_{chunk_size}_{dataset_name}_{split}')
        if os.path.exists(self.save_path):
            self.dataset = load_from_disk(self.save_path)
        else:
            if '2' in self.dataset_name:
                dataset_name = 'wikitext-2-v1'
            else:
                dataset_name = 'wikitext-103-v1'
            raw_dataset = load_dataset('wikitext', dataset_name, cache_dir=cache_dir)[split]
            self.dataset = preprocess_datasets(raw_dataset, tokenizer, chunk_size, num_proc=10)
            self.dataset.save_to_disk(self.save_path)
        self._load_to_device()

    # load text into gpu memory
    def _load_to_device(self):
        save_path = os.path.join(self.cache_dir, f'gptneox_toks_{self.chunk_size}_{self.dataset_name}_{self.split}.pt')
        if not os.path.exists(save_path):
            texts = []
            for ix in tqdm(range(len(self.dataset)), desc='loading dataset'):
                input_ids = torch.tensor(self.dataset[ix]['input_ids'], device=self.device) 
                texts.append(input_ids)
            self.texts = torch.stack(texts)
            torch.save(self.texts, save_path)
        else:
            self.texts = torch.load(save_path, map_location=self.device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ix):
        if self.texts is not None:
            text_ixs = self.texts[ix]
        else:
            ex = self.dataset[ix]
            text_ixs = torch.tensor(ex['input_ids'])
        # for LM dataset, label disagrees by one step
        return text_ixs[:-1], text_ixs[1:]


# from https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb
def preprocess_datasets(dsets, tokenizer, chunk_size, num_proc=1):
    def filter_tokens(examples):
        result = {'text': []}
        for ex in examples['text']:
            # lowercase
            res = ex.lower()

            if res:
                # replace unk token
                res.replace('<unk>', 'UNK')

                # remove symbols
                res = [c for c in res if c.isalnum() or c.isspace()]
                res = ''.join(res)

                res.replace('UNK', '[UNK]') # match bert tokenizer unk token
            result['text'].append(res)
    dsets = dsets.map(filter_tokens, batched=True, num_proc=num_proc)

    def tokenize_fn(examples):
        return tokenizer(examples['text'])

    dsets = dsets.map(tokenize_fn, batched=True, num_proc=num_proc, remove_columns=['text'])
    block_size = chunk_size + 1

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result
    dsets = dsets.map(group_texts, batched=True, num_proc=num_proc)

    return dsets
