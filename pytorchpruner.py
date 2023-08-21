# pruning using pytorch API 
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
import torch.nn.utils.prune as prune
assert torch.cuda.is_available()


#model
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet56", choices=["resnet20", "resnet32", "resnet44", "resnet56",
    "mobilenetv2_x0_5", "mobilenetv2_x0_75", "mobilenetv2_x1_0", "mobilenetv2_x1_4","vgg11_bn", "vgg13_bn",  "vgg16_bn",  "vgg19_bn",
    "repvgg_a0", "repvgg_a1", "repvgg_a2"], help="model to prune")
# dataset
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="dataset to train on")
# arguments 
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
# pruning
parser.add_argument("--method", type=str, default=None, choices=["random", "magnitude_local", "magnitude_global"], help="pruning method")
parser.add_argument("--compression-ratio", type=float, default=None, help="compression ratio")
args = parser.parse_args()


config = {
    "model": args.model,
    "dataset": args.dataset,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "lr": args.lr,
    "method": args.method,
    "compression_ratio": args.compression_ratio,
}

# num classes 
if args.dataset == "cifar10":
    num_classes = 10
elif args.dataset == "cifar100":
    num_classes = 100

run = wandb.init(project=f"pruning_{args.model}_on_{args.dataset}_pytorchAPI", config=config)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = get_model(config["model"], config["dataset"]).to(device)


# Fixer le seed pour la reproductibilité
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# dataloader pour cifar10 et cifar100
NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    'cifar100': dict( mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761)),
    }
image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(**NORMALIZE_DICT[config["dataset"]]),
    ]),
    "test": Compose([
        ToTensor(),
        Normalize(**NORMALIZE_DICT[config["dataset"]]),
    ]),
}
if config["dataset"] == "cifar10":
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(root="data/cifar10", train=(split == "train"), download=True, transform=transforms[split])
    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(dataset[split], batch_size=config["batch_size"], shuffle=(split == 'train'), num_workers=0, pin_memory=True)

elif config["dataset"] == "cifar100":
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR100(root="data/cifar100", train=(split == "train"), download=True, transform=transforms[split])
    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(dataset[split], batch_size=config["batch_size"], shuffle=(split == 'train'), num_workers=0, pin_memory=True)

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
    save_only_state_dict=False,
) -> None:

    optimizer = torch.optim.SGD(model.parameters(
    ), lr=lr, momentum=0.9, weight_decay=weight_decay if pruner is None else 0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,80], gamma=0.1)
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
        print(
            f'Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        # log les valeurs dans wandb
        wandb.log({"val_acc": acc, "val_loss": val_loss,
                  "lr": optimizer.param_groups[0]["lr"]})

        if best_acc < acc:
            best_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_acc = acc
        # Update LR scheduler
        scheduler.step()
    model.load_state_dict(best_checkpoint['state_dict'])
    if save:
        # on veut sauvegarder le meilleur modèle
        path = os.path.join(os.getcwd(), "results", save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if save_only_state_dict:
            torch.save(model.state_dict(), path)
        else:
            torch.save(model, path)     
    print(f'Best val acc: {best_acc:.2f}')




def remove_param(model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try: prune.remove(module, "weight")
            except: pass
            try: prune.remove(module, "bias")
            except: pass
        elif isinstance(module, torch.nn.Linear):
            try: prune.remove(module, "weight")
            except: pass
            try: prune.remove(module, "bias")
            except: pass
    


def main():
    amount_to_prune = 1 - (1 / config["compression_ratio"])

    if config["method"] == 'magnitude_global':
        # global pruning l1 norm
        parameters_to_prune= []
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
            elif isinstance(module, torch.nn.Linear) and module.out_features != num_classes:
                parameters_to_prune.append((module, 'weight'))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount_to_prune)

    elif config["method"] == 'magnitude_local':
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name="weight", amount=amount_to_prune)
            elif isinstance(module, torch.nn.Linear) and module.out_features != num_classes:
                prune.l1_unstructured(module, name="weight", amount=amount_to_prune)

    elif config["method"] == 'random':
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.random_unstructured(module, name="weight", amount=amount_to_prune)
            elif isinstance(module, torch.nn.Linear) and module.out_features != num_classes:
                prune.random_unstructured(module, name="weight", amount=amount_to_prune)

    path = f'results/{config["model"]}_compression_{config["compression_ratio"]}_{config["dataset"]}_{args.method}.pth'
    train(model, dataloader["train"], dataloader["test"], epochs=config["epochs"], lr=config["lr"])

    # remove masks
    remove_param(model)
    
    # save model
    torch.save(model, path)

    example_input = torch.rand(1, 3, 32, 32).to(device)
    end_macs, end_params = tp.utils.count_ops_and_params(model, example_input)
    end_acc, end_loss = evaluate(model, dataloader["test"])
    # log les valeurs dans wandb
    wandb.run.summary["best_acc"] = end_acc
    wandb.run.summary["best_loss"] = end_loss
    wandb.run.summary["end macs (M)"] = end_macs/1e6
    wandb.run.summary["end num_params (M)"] = end_params/1e6
    wandb.run.summary["size (MB)"] = get_model_size(model)/8e6


if __name__ == "__main__":
    main()