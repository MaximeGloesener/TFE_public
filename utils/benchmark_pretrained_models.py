from registry import get_model
from torch2trt import torch2trt
import torch_pruning as tp 
from utils.benchmark import *

models = [
    "resnet20", "resnet32", "resnet44", "resnet56",
    "mobilenetv2_x0_5", "mobilenetv2_x0_75", "mobilenetv2_x1_0", "mobilenetv2_x1_4",
     "vgg11_bn", "vgg13_bn",  "vgg16_bn",  "vgg19_bn",
    "repvgg_a0", "repvgg_a1", "repvgg_a2"
]


datasets = ["cifar100"]




from torchvision.datasets import *
from torchvision.transforms import *
from torch.utils.data import DataLoader
# dataloader pour cifar10 et cifar100
NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'cifar100': dict( mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761)),
    }
image_size = 32
transforms_cifar10 = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(**NORMALIZE_DICT['cifar10']),
    ]),
    "test": Compose([
        ToTensor(),
        Normalize(**NORMALIZE_DICT['cifar10']),
    ]),
}
transforms_cifar100 = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(**NORMALIZE_DICT['cifar100']),
    ]),
    "test": Compose([
        ToTensor(),
        Normalize(**NORMALIZE_DICT['cifar100']),
    ]),
}

dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(root="data/cifar10", train=(split == "train"), download=True, transform=transforms_cifar10[split])
dataloaderc10 = {}
for split in ['train', 'test']:
    dataloaderc10[split] = DataLoader(dataset[split], batch_size=512, shuffle=(split == 'train'), num_workers=0, pin_memory=True)

dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR100(root="data/cifar100", train=(split == "train"), download=True, transform=transforms_cifar100[split])
dataloaderc100 = {}
for split in ['train', 'test']:
    dataloaderc100[split] = DataLoader(dataset[split], batch_size=512, shuffle=(split == 'train'), num_workers=0, pin_memory=True)

import torch 
from tqdm import tqdm
import torch.nn as nn
device = torch.device('cuda')
import torch.nn.functional as F
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


example_input = torch.randn(64, 3, 32, 32).cuda()

for model_name in models:
    for dataset in datasets:
        print(f'model name = {model_name} / dataset = {dataset}')
        model = get_model(model_name, dataset)
        model.eval()
        measure_latency_gpu(model, example_input,n_warmup=80,n_test=500)
"""        if dataset == 'cifar10':
            macs, param = tp.utils.count_ops_and_params(model, example_input)
            print("macs", macs, "param", param)
            macs = get_model_macs(model, example_input)
            print("macs", macs/1e6)
            param = get_num_parameters(model)
            print("param",param/1e6)
            size = get_model_size(model)
            print("size", size/(8*1000*1000))
            acc, loss = evaluate(model, dataloaderc10['test'])
            print(f' accuracy = {acc:.2f} %')

"""