import torch.nn.functional as F
from utils.benchmark import *
import torch
import os
from models.vgg_tiny import VGG
# Imports
import copy
import random
import numpy as np
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
assert torch.cuda.is_available()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Fixer le seed pour la reproductibilité
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Datas
# normalize data pour resnet56
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
        batch_size=512,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True,
    )
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
# training loop
def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: int,
    # for pruning
    weight_decay=5e-4,
    pruner=None,   
    callbacks=None,
    save=None, 
) -> None:

  
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay if pruner is None else 0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1
    best_checkpoint = dict()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset the gradients (from the last iteration)
            optimizer.zero_grad()

            # Forward inference
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward propagation
            loss.backward()

            # Pruner regularize for sparsity learning
            if pruner is not None:
                pruner.regularize(model)

            # Update optimizer
            optimizer.step()
           

            if callbacks is not None:
                for callback in callbacks:
                    callback()

        acc, val_loss = evaluate(model, test_loader)
        print(f'Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        if best_acc < acc:
            best_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_acc = acc
        scheduler.step()
    model.load_state_dict(best_checkpoint['state_dict'])
    
    if save:
        # on veut sauvegarder le meilleur modèle 
        path = os.path.join(os.getcwd(), "results", save) 
        os.makedirs(os.path.dirname(path), exist_ok=True)   
        torch.save(model, path)
    print(f'Best val acc: {best_acc:.2f}')



import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib

# add quantizer/dequantizer layer to the model
quant_modules.initialize()


checkpoint = torch.load("models/pretrained_state_dicts_cifar10/vgg.cifar.pretrained.pth", map_location="cpu")
model = VGG().cuda()
model.load_state_dict(checkpoint["state_dict"])
acc, loss = evaluate(model, dataloader['test'])
print(f"Pretrained model accuracy: {acc:.2f} | loss: {loss:.4f}")

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

# It is a bit slow since we collect histograms on CPU
with torch.no_grad():
    collect_stats(model, dataloader["train"], num_batches=50)
    compute_amax(model, method="percentile", percentile=99.99)

with torch.no_grad():
    compute_amax(model, method="percentile", percentile=99.9)
    acc, loss = evaluate(model, dataloader["test"])
    print(acc, loss)


with torch.no_grad():
    for method in ["mse", "entropy"]:
        print(F"{method} calibration")
        compute_amax(model, method=method)
        acc, loss = evaluate(model, dataloader["test"])
        print(acc, loss)


acc, loss = evaluate(model, dataloader["test"])
print(acc, loss)

train(model, dataloader["train"], dataloader["test"], epochs=10, lr=0.1)

acc, loss = evaluate(model, dataloader["test"])
print(acc, loss)

"""# Export the model to ONNX
from pytorch_quantization import nn as quant_nn
quant_nn.TensorQuantizer.use_fb_fake_quant = True

torch.onnx.export(model, dummy_input, "quant_vgg3.onnx", verbose=True)"""