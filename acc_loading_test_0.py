import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import os


class NumberDataset(Dataset):
    def __init__(self, size=20):
        self.data = torch.arange(size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


batch_size = 5

dataset = NumberDataset()
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)


for epoch in range(1):
    for i, batch in enumerate(data_loader):

        inputs = batch.float().unsqueeze(1)
        targets = inputs * 2
        
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        print(f'Batch: {batch.tolist()}')