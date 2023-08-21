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


# parser 
parser = argparse.ArgumentParser()

#model
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
parser.add_argument("--method", type=str, default=None, choices=["random", "l1", "lamp", "slim", "group_norm", "group_sl"], help="pruning method")
parser.add_argument("--speed-up", type=float, default=None, help="speed up factor")
parser.add_argument("--compression-ratio", type=float, default=None, help="compression ratio")
parser.add_argument("--global-pruning", action="store_true", default=False)
parser.add_argument("--reg", type=float, default=1e-5)
parser.add_argument("--sl-total-epochs", type=int, default=100, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
parser.add_argument("--iterative-steps", default=400, type=int)
parser.add_argument("--max-sparsity", type=float, default=1.0)
# kd 
parser.add_argument("--kd", action="store_true", default=False)
parser.add_argument("--alpha-kd", type=float, default=0.9, help='alpha for kd loss')
parser.add_argument("--temperature-kd", type=float, default=4, help='temperature for kd loss')
args = parser.parse_args()

# num classes 
if args.dataset == "cifar10":
    num_classes = 10
elif args.dataset == "cifar100":
    num_classes = 100


# wandb config
config = {
    "model": args.model,
    "num_classes": num_classes,
    "random_seed": 42,
    "dataset": args.dataset,
    "batch_size": args.batch_size,
    "optimizer": "SGD",
    "lr": args.lr,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "MultiStepLR - decay 60 80",
    "epochs_long_finetuning": args.epochs,
    "epochs_short_finetuning": 10,
    "loss": "CrossEntropyLoss",
    "pruner": args.method,
    "speed_up": args.speed_up,
    "compression_ratio": args.compression_ratio,
    "global_pruning": args.global_pruning,
    "reg": args.reg,
    "sl_total_epochs": args.sl_total_epochs,
    "sl_lr": args.sl_lr,
    "iterative_steps": args.iterative_steps,
}

run = wandb.init(project=f"end_pruning_{args.model}_on_{args.dataset}", config=config)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = get_model(config["model"], config["dataset"]).to(device)
model_teacher = copy.deepcopy(model) # save teacher model for kd


# Fixer le seed pour la reproductibilité
random.seed(config["random_seed"])
np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])


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
    weight_decay=config["weight_decay"],
    pruner=None,
    callbacks=None,
    save=None,
    save_only_state_dict=False,
) -> None:

    optimizer = torch.optim.SGD(model.parameters(
    ), lr=lr, momentum=config["momentum"], weight_decay=weight_decay if pruner is None else 0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,100], gamma=0.1)
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

# training loop
def train_kd(
    model_student: nn.Module,
    model_teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: int,
    alpha: float,
    temperature:int,
    weight_decay=5e-4,
    callbacks=None,
    save=None,
    save_only_state_dict=False,
) -> None:

    optimizer = torch.optim.SGD(model_student.parameters(
    ), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,80], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1
    best_checkpoint = dict()

  
    for epoch in range(epochs):
        model_student.train()
        model_teacher.train()
        for inputs, targets in tqdm(train_loader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset the gradients (from the last iteration)
            optimizer.zero_grad()

            # Forward inference
            out_student = model_student(inputs)
            out_teacher = model_teacher(inputs)


            # kd loss
            predict_student = F.log_softmax(out_student / temperature, dim=1)
            predict_teacher = F.softmax(out_teacher / temperature, dim=1)
            loss = nn.KLDivLoss(reduction="batchmean")(predict_student, predict_teacher) * (alpha * temperature * temperature) + criterion(out_student, targets) * (1-alpha)
            
            loss.backward()


            # Update optimizer
            optimizer.step()

            if callbacks is not None:
                for callback in callbacks:
                    callback()

        acc, val_loss = evaluate(model_student, test_loader)
        print(
            f'Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        # log les valeurs dans wandb
        wandb.log({"val_acc": acc, "val_loss": val_loss,
                  "lr": optimizer.param_groups[0]["lr"]})
        if best_acc < acc:
            best_checkpoint['state_dict'] = copy.deepcopy(model_student.state_dict())
            best_acc = acc
        # Update LR scheduler
        scheduler.step()
    model_student.load_state_dict(best_checkpoint['state_dict'])
    if save:
        # on veut sauvegarder le meilleur modèle
        path = os.path.join(os.getcwd(), "results", save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if save_only_state_dict:
            torch.save(model_student.state_dict(), path)
        else:
            torch.save(model_student, path)     
    print(f'Best val acc: {best_acc:.2f}')

# Pruner
# définir le nbre de classses => évite de pruner la dernière couche
def get_pruner(model, example_input):
    args.sparsity_learning = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.BNScalePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        args.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError

    unwrapped_parameters = []
    ignored_layers = []
    ch_sparsity_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == num_classes:
            ignored_layers.append(m)
    

    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_input,
        importance=imp,
        iterative_steps=args.iterative_steps,
        ch_sparsity=1,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=args.max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner

# pruning jusqu'à atteindre le speed up voulu
def progressive_pruning_speedup(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(
        model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        # print(current_speed_up)
        if pruner.current_step == pruner.iterative_steps:
            break
    return current_speed_up


# pruning jusqu'à atteindre le ratio de compression voulu
def progressive_pruning_compression_ratio(pruner, model, compression_ratio, example_inputs):
    # compression ratio défini par taille initiale / taille finale
    model.eval()
    _, base_params = tp.utils.count_ops_and_params(
        model, example_inputs=example_inputs)
    current_compression_ratio = 1
    while current_compression_ratio < compression_ratio:
        pruner.step(interactive=False)
        _, pruned_params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs)
        current_compression_ratio = float(base_params) / pruned_params
        if pruner.current_step == pruner.iterative_steps:
            break
        # print(current_compression_ratio)
    return current_compression_ratio


# pruning jusqu'à atteindre le speed up voulu + fine tuning après chaque étape de pruning
def progressive_pruning_fine_tuning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(
        model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        # print(current_speed_up)
        # après avoir pruné : fine tuning
        train(model, dataloader["train"],
              dataloader["test"], epochs=config["epochs_short_finetuning"], lr=0.01)
        if pruner.current_step == pruner.iterative_steps:
            break
    return current_speed_up


def main():
    # Avant pruning
    example_input = torch.rand(1, 3, 32, 32).to(device)
    start_macs, start_params = tp.utils.count_ops_and_params(model, example_input)
    start_acc, start_loss = evaluate(model, dataloader["test"])
    print('----- Avant pruning -----')
    print(f'Nombre de MACs = {start_macs/1e6:.3f} M')
    print(f'Nombre de paramètres = {start_params/1e6:.3f} M')
    print(f'Précision = {start_acc:.2f} %')
    print(f'Loss = {start_loss:.3f}')
    print('')
    wandb.run.summary["start_macs (M)"] = f'{start_macs/1e6:.3f}'
    wandb.run.summary["start_params (M)"] = f'{start_params/1e6:.3f}'
    wandb.run.summary["start_acc (%)"] = f'{start_acc:.2f}'
    wandb.run.summary["start_loss"] = f'{start_loss:.3f}'

    # speed_up = config["speed_up"]
    compression_ratio = config["compression_ratio"]
    pruner = get_pruner(model, example_input)
    reg_path = f"{config['model']}_{args.method}_{args.dataset}.pth"
    if args.sparsity_learning:
        print('loading regularized model')
        # si le modèle régularisé n'existe pas, on le crée
        if not os.path.exists(f"results/{reg_path}"):
        # (for faster experiments we don't regularize each time, we just load the regularized model)
            print('----- Regularizing -----')
            train(
                model,
                train_loader=dataloader["train"],
                test_loader=dataloader["test"],
                epochs=args.sl_total_epochs,
                lr=args.sl_lr,
                pruner=pruner,
                save=reg_path,
                save_only_state_dict=True,
            )
        model.load_state_dict(torch.load(f"results/{reg_path}"))
        model.cuda()

    #progressive_pruning(pruner, model, speed_up, example_input)
    progressive_pruning_compression_ratio(pruner, model, compression_ratio, example_input)
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(
        model, example_input)
    pruned_acc, pruned_loss = evaluate(model, dataloader["test"])
    print('----- Après pruning -----')
    print(f'Nombre de MACs = {pruned_macs/1e6:.3f} M')
    print(f'Nombre de paramètres = {pruned_params/1e6:.3f} M')
    print(f'Précision = {pruned_acc:.2f} %')
    print(f'Loss = {pruned_loss:.3f}')
    print('')

    # Results
    print('----- Results before fine tuning -----')
    print(f'Params: {start_params/1e6:.2f} M => {pruned_params/1e6:.2f} M')
    print(f'MACs: {start_macs/1e6:.2f} M => {pruned_macs/1e6:.2f} M')
    print(f'Accuracy: {start_acc:.2f} % => {pruned_acc:.2f} %')
    print(f'Loss: {start_loss:.2f} => {pruned_loss:.2f}')
    print('')

    # Fine tuning
    print('----- Fine tuning -----')
    path = f'{config["model"]}_compression_{config["compression_ratio"]}_{config["dataset"]}_{args.method}.pth'
    if args.kd:
        print('----- Knowledge distillation -----')
        train_kd(model, model_teacher, dataloader["train"], dataloader["test"],
                    epochs=config["epochs_long_finetuning"], lr=config["lr"], alpha=args.alpha_kd, temperature=args.temperature_kd, save=path)
    else:
        train(model, dataloader["train"], dataloader["test"],
        epochs=config["epochs_long_finetuning"], lr=config["lr"], save=path)

    # Post fine tuning
    end_macs, end_params = tp.utils.count_ops_and_params(model, example_input)
    end_acc, end_loss = evaluate(model, dataloader["test"])
    print('----- Results after fine tuning -----')
    print(f'Params: {start_params/1e6:.2f} M => {end_params/1e6:.2f} M')
    print(f'MACs: {start_macs/1e6:.2f} M => {end_macs/1e6:.2f} M')
    print(f'Accuracy: {start_acc:.2f} % => {end_acc:.2f} %')
    print(f'Loss: {start_loss:.2f} => {end_loss:.2f}')
    # log les valeurs dans wandb
    wandb.run.summary["best_acc"] = end_acc
    wandb.run.summary["best_loss"] = end_loss
    wandb.run.summary["end macs (M)"] = end_macs/1e6
    wandb.run.summary["end num_params (M)"] = end_params/1e6
    wandb.run.summary["size (MB)"] = get_model_size(model)/8e6

if __name__ == "__main__":
    main()