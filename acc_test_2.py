import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from deepspeed.ops.adam import FusedAdam
import torch.distributed as dist
import os


class NumberDataset(Dataset):
    def __init__(self, size=20):
        self.data = torch.arange(size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


batch_size = 5
dataset_size = 30
zero_stage = 1
gradient_accumulation_steps = 2
world_size = int(os.environ.get('WORLD_SIZE', 1))

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

accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)



dataset = NumberDataset(dataset_size)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = nn.Linear(1, 1)
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = FusedAdam(model.parameters(), lr=0.001)


model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

for epoch in range(1):
    for i, batch in enumerate(data_loader):

        inputs = batch.float().unsqueeze(1)
        targets = inputs * 2
        
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        
        accelerator.backward(loss) # gradient scaling etc handled with respect to grad acc steps

        if (i+1) % accelerator.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        
        print(f"Process {accelerator.process_index}, Batch: {batch.tolist()}")



