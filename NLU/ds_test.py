from datasets import load_dataset, load_metric
from transformers import PreTrainedTokenizerFast,AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
# from src.model_classify import GPT, GPTConfig
import torch
import datasets
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from src.spikingjelly.clock_driven import functional
from transformers import DataCollatorWithPadding
import os
from functools import partial
from tokenizers import Tokenizer

DATA_PATH = '/workspace/data_copy'
BATCH_SIZE = 2

class ParenFreeFunc:
    def __init__(self, func):
        self.func = func

    def __ror__(self, arg):
        return self.func(arg)

_pathfunc = partial(os.path.join, DATA_PATH)
pathfunc = ParenFreeFunc(_pathfunc)

filename = 'Wikipedia/jsonl/wiki_0072.jsonl.gz' | pathfunc

vocab_size = 32768
tokenizer = Tokenizer.from_file(f"/workspace/spiking_workspace/BPE_byte_fallback_{vocab_size}.json")


def tokenize(batch):
    return tokenizer(batch["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataset = datasets.load_dataset(
    'json',
    data_files=[filename],
    streaming=True,
)['train'].map(tokenize, batched=True, batch_size=BATCH_SIZE)



def collate_fn(examples):

    examples = tokenizer.pad(
        examples,
        padding=True,
        max_length=None,
    )

    new_batch_data = []
    new_batch_label = []

    for i in range(len(examples['input_ids'])):
        new_batch_data.append(torch.tensor(examples['input_ids'][i]))
        new_batch_label.append(torch.tensor(examples['label'][i], dtype=torch.long))
    data = torch.stack(new_batch_data, dim=0)
    label = torch.stack(new_batch_label, dim=0)

    return data, label


train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

