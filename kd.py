import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
import torch.nn.functional as F
import random
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
def train_kd(
    model_student: nn.Module,
    model_teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: int,
    weight_decay=5e-4,
    callbacks=None,
    save=None,
    save_only_state_dict=False,
) -> None:

    optimizer = torch.optim.SGD(model_student.parameters(
    ), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.1)
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
            kd_T = 4
            predict_student = F.log_softmax(out_student / kd_T, dim=1)
            predict_teacher = F.softmax(out_teacher / kd_T, dim=1)
            alpha = 0.9
            loss = nn.KLDivLoss()(predict_student, predict_teacher) * (alpha * kd_T * kd_T) + criterion(out_student, targets) * (1-alpha)
            
            loss.backward()


            # Update optimizer
            optimizer.step()

            if callbacks is not None:
                for callback in callbacks:
                    callback()

        acc, val_loss = evaluate(model_student, test_loader)
        print(
            f'Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
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

# Fixer le seed pour la reproductibilité
import numpy as np
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
        Normalize(**NORMALIZE_DICT['cifar10']),
    ]),
    "test": Compose([
        ToTensor(),
        Normalize(**NORMALIZE_DICT['cifar10']),
    ]),
}
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(root="data/cifar10", train=(split == "train"), download=True, transform=transforms[split])
dataloader = {}
for split in ['train', 'test']:
    dataloader[split] = DataLoader(dataset[split], batch_size=512, shuffle=(split == 'train'), num_workers=0, pin_memory=True)

from models.resnet import resnet56

checkpoint = torch.load("models/pretrained_state_dicts_cifar10/cifar10_resnet56.pt", map_location="cpu")
teacher_model = resnet56(10).to(device)
teacher_model.load_state_dict(checkpoint)
acc, loss = evaluate(teacher_model, dataloader["test"])
print(f"Teacher model: acc: {acc:.2f} loss: {loss:.4f}")


# test student model
print('student model')
student_model = torch.load("results/repvgg_a0_compression_64.0_cifar10_group_sl.pth", map_location="cpu").to(device)
student_model.eval()
acc, loss = evaluate(student_model, dataloader["test"])
print(f"Student model: acc: {acc:.2f} loss: {loss:.4f}")

# train student model
print('train student model')
train_kd(student_model, teacher_model, dataloader["train"], dataloader["test"], epochs=100, lr=0.001, save="kdtest.pt")