import torch
import numpy as np 
import torch.nn.functional as F
from utils.benchmark import *
import torch
from models.vgg_tiny import VGG
import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
assert torch.cuda.is_available()
from models.vgg import vgg16_bn
from utils.benchmark import measure_latency_gpu
from models.resnet import *
from models.vgg import *
from models.repvgg import *

model_name = "repvgg_a1"
checkpoint = torch.load("models/pretrained_state_dicts_cifar10/cifar10_repvgg_a1.pt", map_location="cpu")
model = repvgg_a1(10)
model.load_state_dict(checkpoint)
model.eval()
device = torch.device('cuda')
model.to(device)


example_input = torch.randn(1, 3, 32, 32).cuda()



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
   'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
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
        batch_size=1,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True,
    )





from torch2trt import torch2trt

model = torch.load('results_experiments/2023-08-15_10-04-47/kd_model.pth').cuda()
measure_latency_gpu(model, example_input.cuda())

calib_dataset = list()
for i, img in enumerate(dataloader["train"]):
    calib_dataset.extend(torch.randn(1, 3, 32, 32).cuda())
    if i ==1:
        break

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model,[example_input], fp16_mode=True, int8_mode=True, int8_calib_dataset=calib_dataset, max_batch_size=128)

print('modèle optimisé')
measure_latency_gpu(model_trt, example_input.cuda())
acc ,loss = evaluate(model_trt, dataloader['test'])
print(acc, loss)
# acc, loss = evaluate(model, dataloader['test'])
# print(acc, loss)

path = "results_quant224/"
torch.save(model_trt.state_dict(), path+f"abtestt.pth")