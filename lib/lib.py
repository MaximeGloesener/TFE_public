"""
Fichier principal de la librairie pour faire pruning/KD/quantization
""" 

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
import logging 
from datetime import datetime
from torch2trt import torch2trt
assert torch.cuda.is_available(), "Cuda Not Available!"

# Device
device = torch.device("cuda")

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
        logging.info(
            f'Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        
        # log les valeurs dans wandb
        if wandb.run is not None:
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
        # path = os.path.join(os.getcwd(), "results", save)
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        if save_only_state_dict:
            torch.save(model.state_dict(), save)
        else:
            torch.save(model, save)     
    logging.info(f'Best val acc: {best_acc:.2f}')


# Pruner
# définir le nbre de classses => évite de pruner la dernière couche
def get_pruner(model, example_input, num_classes):
    sparsity_learning = True
    imp = tp.importance.GroupNormImportance(p=2)
    pruner_entry = partial(tp.pruner.GroupNormPruner, reg=1e-5, global_pruning=True)


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
        iterative_steps=400,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=0.75,
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
        # print(current_compression_ratio)
    return current_compression_ratio

# training loop
def train_kd(
    model_student: nn.Module,
    model_teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: int,
    temperature: int,
    alpha: float,
    weight_decay=5e-4,
    callbacks=None,
    save=None,
    save_only_state_dict=False,
) -> None:

    optimizer = torch.optim.SGD(model_student.parameters(
    ), lr=lr, momentum=0.9, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,80,110], gamma=0.1)
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
        logging.info(
            f'KD - Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # log les valeurs dans wandb
        if wandb.run is not None:
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
        if save_only_state_dict:
            torch.save(model_student.state_dict(), save)
        else:
            torch.save(model_student, save)     
    logging.info(f'Best val acc after KD: {best_acc:.2f}')     





def get_compression_ratio_and_bitwidth_from_compression_ratio(compression_ratio):
    if compression_ratio <=2:
        return compression_ratio, 32
    elif  2 < compression_ratio <= 4:
        return compression_ratio/2, 16
    else:
        return compression_ratio/4, 8 


def get_speed_up_and_bitwidth_from_speed_up(speed_up):
    if speed_up <=2:
        return speed_up, 32
    elif  2 < speed_up <= 4:
        return speed_up/2, 16
    else:
        return speed_up/4, 8 



def optimize(model, 
            traindataloader,
            testdataloader, 
            example_input, 
            num_classes,
            epochs=120,
            lr=0.01,
            temperature=4,
            alpha=0.9, 
            compression_ratio=2, 
            speed_up=None, 
            bitwidth=8, 
            wandb_project=None, 
            random_seed=42):
    # on  veut log tous les résultats intermédiaires (modèle régularizé/modèle pruné) dans un dossier results
    if not os.path.exists("results_experiments"):
        os.makedirs("results_experiments")
    # subfolder for each run
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"results_experiments/{run_id}")
    base_path = f"results_experiments/{run_id}"
    # logging
    logging.basicConfig(filename=f"{base_path}/log.txt", level=logging.INFO)
    logging.info(f"Run ID: {run_id}")
    logging.info(f"Model: {model.__class__.__name__}")

    example_input = example_input.to(device)
    
    if wandb_project:
        run = wandb.init(project=wandb_project)
        logging.info("Wandb initialized")
        logging.info(f"Wandb project: {wandb_project}")
        

    # Fixer le seed pour la reproductibilité
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Copy the initial model for KD 
    model_teacher = copy.deepcopy(model)

    # Avant pruning
    start_macs, start_params = tp.utils.count_ops_and_params(model, example_input)
    start_acc, start_loss = evaluate(model, testdataloader)
    logging.info(' ----- Initial Model: -----')
    logging.info(f'Number of MACs = {start_macs/1e6:.3f} M')
    logging.info(f'Number of Parameters = {start_params/1e6:.3f} M')
    logging.info(f'Accuracy = {start_acc:.2f} %')
    logging.info(f'Loss = {start_loss:.3f}')
    logging.info(' ---------------------------')
    if wandb_project:
        wandb.run.summary["start_macs (M)"] = f'{start_macs/1e6:.3f}'
        wandb.run.summary["start_params (M)"] = f'{start_params/1e6:.3f}'
        wandb.run.summary["start_acc (%)"] = f'{start_acc:.2f}'
        wandb.run.summary["start_loss"] = f'{start_loss:.3f}'

    if compression_ratio:
        compression_ratio, bitwidth = get_compression_ratio_and_bitwidth_from_compression_ratio(compression_ratio)
    if speed_up:
        speed_up, bitwidth = get_speed_up_and_bitwidth_from_speed_up(speed_up)

    pruner = get_pruner(model, example_input, num_classes)

    reg_path = f"{base_path}/{model.__class__.__name__}_regularized.pth"
    logging.info('Regularizing the model...')
    # si le modèle régularisé n'existe pas, on le régularize
    if not os.path.exists(f"{reg_path}"):
    # (for faster experiments we don't regularize each time, we just load the regularized model)
        train(
            model,
            train_loader=traindataloader,
            test_loader=testdataloader,
            epochs=100,
            lr=0.001,
            pruner=pruner,
            save=reg_path,
            save_only_state_dict=True,
        )
    logging.info('Regularization done')
    model.load_state_dict(torch.load(f"{reg_path}"))
    model.cuda()

    logging.info('----- Pruning -----')
    if compression_ratio:
        progressive_pruning_compression_ratio(pruner, model, compression_ratio, example_input)
    else:
        progressive_pruning_speedup(pruner, model, speed_up, example_input)

    # Fine tuning
    logging.info('----- Fine tuning with KD -----')
    train_kd(model, model_teacher, traindataloader, testdataloader, epochs=epochs, lr=lr, temperature=temperature, alpha=alpha,save=f'{base_path}/kd_model.pth')
    # train(model, traindataloader, testdataloader, epochs=epochs, lr=lr, save=f'{base_path}/kd_model.pth')
    # Post fine tuning
    end_macs, end_params = tp.utils.count_ops_and_params(model, example_input)
    end_acc, end_loss = evaluate(model, testdataloader)
    logging.info('----- Results after fine tuning -----')
    logging.info(f'Number of Parameters: {start_params/1e6:.2f} M => {end_params/1e6:.2f} M')
    logging.info(f'MACs: {start_macs/1e6:.2f} M => {end_macs/1e6:.2f} M')
    logging.info(f'Accuracy: {start_acc:.2f} % => {end_acc:.2f} %')
    logging.info(f'Loss: {start_loss:.2f} => {end_loss:.2f}')
    if wandb_project:
        # log les valeurs dans wandb
        wandb.run.summary["best_acc"] = end_acc
        wandb.run.summary["best_loss"] = end_loss
        wandb.run.summary["end macs (M)"] = end_macs/1e6
        wandb.run.summary["end num_params (M)"] = end_params/1e6
        wandb.run.summary["size (MB)"] = get_model_size(model)/8e6

    # Quantization part 
    logging.info('----- Quantization -----')
    # free cache memory 
    torch.cuda.empty_cache()
    # if user want to choose the bitwidth
    if bitwidth == 8:
        logging.info('Calibrating on train dataset...')
        calib_dataset = list()
        for i, img in enumerate(traindataloader):
            calib_dataset.extend(img[0])
            if i == 3000:
                break
        model_trt = torch2trt(model,[example_input], fp16_mode=True, int8_mode=True, int8_calib_dataset=calib_dataset, max_batch_size=128)
        compression_ratio_quant = 4
    elif bitwidth == 16:
        model_trt = torch2trt(model,[example_input], fp16_mode=True, max_batch_size=128)
        compression_ratio_quant = 2
    else: # run with fp32 if not quantization -> speed up from tensorrt inference engine
        model_trt = torch2trt(model,[example_input], max_batch_size=128)
        bitwidth = 32
        compression_ratio_quant = 1
    logging.info(f"Final Compression Ratio: {start_params/end_params*compression_ratio_quant:.2f}") 
    logging.info(f"Bit width: {bitwidth}")
    torch.save(model_trt.state_dict(), f'{base_path}/model_trt.pth')
    
    return model_trt