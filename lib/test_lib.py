# pruner retrain and log everything in wandb
# Imports
import wandb
import torch.nn.functional as F
from utils.benchmark import *
import torch
import torch_pruning as tp
import os
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
import argparse
from functools import partial
from registry import get_model
assert torch.cuda.is_available()


device = torch.device('cuda')


# Evaluation loop
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    verbose=True,
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0
    loss = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inference
        outputs = model(inputs)
        # Calculate loss
        loss += F.cross_entropy(outputs, targets, reduction="sum")
        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)
        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()
    return (num_correct / num_samples * 100).item(), (loss / num_samples).item()

# Datas
NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
    }
image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize( **NORMALIZE_DICT['cifar10']),
    ]),
    "test": Compose([
        ToTensor(),
        Normalize( **NORMALIZE_DICT['cifar10']),
    ]),
}



# récupérer les jeux de données d'entrainement et de test
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(
        root="data/cifar10",
        train=(split == "train"),
        download=True,
        transform=transforms[split],
    )
dataloader = {}
for split in ['train', 'test']:
    dataloader[split] = DataLoader(
        dataset[split],
        batch_size=256,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True,
    )

import wandb
from lib import optimize
from registry import get_model
parser = argparse.ArgumentParser()

run = wandb.init(project="end_ALL")

#model
parser.add_argument("--model", type=str)

args = parser.parse_args()
print(args.model)
model = get_model(args.model, "cifar10")


# évaluer le modèle
acc, loss = evaluate(model, dataloader['test'])

# optimiser le modèle
model_optimized = optimize(model, 
                           dataloader['train'], 
                           dataloader['test'], 
                           example_input=torch.randn(1, 3, 32, 32),
                           num_classes= 10)

# évaluer le modèle optimisé
acc_optimized, loss_optimized = evaluate(model_optimized, dataloader['test'])