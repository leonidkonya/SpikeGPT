import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from deepspeed.ops.adam import FusedAdam
import torch.distributed as dist
import os
from functools import partial
from tokenizers import Tokenizer
from types import SimpleNamespace
import numpy as np
from torch.utils.data.sampler import BatchSampler, RandomSampler
import yaml
from datasets import load_from_disk

from src.packing_utils import (
    print_packed_batch,
    SamplePackingBatchSampler,
    collate_fn_base,
    get_packing_stats,
)

from accelerate import DataLoaderConfiguration

# from torch.utils.data.distributed import DistributedSampler


"""
proper distributed sampling attempt

"""


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


use_seedable_sampler = True


config_path = 'config/conf_small_test.yml'
with open(config_path, 'r') as f:
    args = SimpleNamespace(**yaml.safe_load(f))

pretokenized_dir = '/workspace/spiking_workspace/pretokenized_custom/'

data_path = args.data_train

fname = os.path.basename(data_path[0])
fname_stem = fname.split('.')[0]

pretokenized_path = os.path.join(pretokenized_dir, fname_stem)
dataset = load_from_disk(pretokenized_path)


tokenizer = Tokenizer.from_file(args.tokenizer_path)
eod_token_id = tokenizer.token_to_id('[SEP]')
pad_token_id = tokenizer.token_to_id('[PAD]')
ctx_len = args.ctx_len

collate_fn = partial(
    collate_fn_base,
    eod_token_id=eod_token_id,
    pad_token_id=pad_token_id,
    ctx_len=ctx_len,
)


batch_size = 4
dataset_size = 128
zero_stage = 1
gradient_accumulation_steps = 1
world_size = int(os.environ.get('WORLD_SIZE', 1))
packing_group_size = 16

train_batch_size = batch_size * gradient_accumulation_steps * world_size


deepspeed_config = {
    "train_batch_size": train_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "train_micro_batch_size_per_gpu": batch_size,
    "zero_optimization": {
        "stage": zero_stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_scatter": True,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
    },
}


deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=zero_stage,
    hf_ds_config=deepspeed_config,
    gradient_accumulation_steps=gradient_accumulation_steps,
)

dataloader_config = DataLoaderConfiguration(
    use_seedable_sampler=use_seedable_sampler,
)

accelerator = Accelerator(
    deepspeed_plugin=deepspeed_plugin,
    dataloader_config=dataloader_config,
)


dataset = load_from_disk(pretokenized_path)
dataset = dataset.select(range(dataset_size))
dataset.__getitems__ = None


input_ids = dataset.data.column("input_ids")
lengths = np.vectorize(len)(np.array(input_ids, dtype=object))

packing_sampler = SamplePackingBatchSampler(
    batch_size=batch_size,
    ctx_len=args.ctx_len,
    group_size=packing_group_size,
    lengths=lengths,
    sampler=RandomSampler(dataset),
)

data_loader = DataLoader(
    dataset=dataset,
    batch_sampler=packing_sampler,
    collate_fn=collate_fn,
    num_workers=2,
    persistent_workers=True, # don't attempt to delete file handles because it is in use by the dataset itself (arrow)
)


model = nn.Linear(1, 1)
optimizer = FusedAdam(model.parameters(), lr=0.001)


model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

doc_count_total = 0
for i, (x, y) in enumerate(data_loader):
    # print_packed_batch(x, tokenizer, eod_token_id)

    pad, eod, doc_count = get_packing_stats(x, eod_token_id, pad_token_id)
    doc_count_total += doc_count

print(f'doc count total: {doc_count_total}')





# for epoch in range(1):
#     for i, batch in enumerate(data_loader):

#         inputs = batch.float().unsqueeze(1)
#         targets = inputs * 2
        
#         outputs = model(inputs)
#         loss = nn.MSELoss()(outputs, targets)
        
#         accelerator.backward(loss) # gradient scaling etc handled with respect to grad acc steps

#         if (i+1) % accelerator.gradient_accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()
        
        
        # print(f"Process {accelerator.process_index}, Batch: {batch.tolist()}")



