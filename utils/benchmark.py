import torch
from torchprofile import profile_macs
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np


def get_model_macs(model, inputs) -> int:
    # attention le nombre de macs dépend de la dimension de l'entrée 
    # => si on prend un batch en entrée alors MACs >>>> 
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


# sparsity pour chaque couche du modèle
def get_layer_sparsity(model: nn.Module) -> dict:
    print('Sparsité par couche: ')
    for name, param in model.named_parameters():
        if param.dim() > 1:
            print(f'{name}: {get_sparsity(param):.3f}')


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements


def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


# Nombre de paramètres de chaque couche du modèle
def plot_num_parameters_distribution(model):
    num_parameters = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            num_parameters[name] = param.numel()
    fig = plt.figure(figsize=(8, 6))
    plt.grid(axis='y')
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width


# Distribution des poids
def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    nlayers = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nlayers += 1
    fig, axes = plt.subplots((nlayers//2)+1, 2, figsize=(10, 10))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color='blue', alpha=0.5)
            else:
                ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                        color='blue', alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


@torch.no_grad()
def measure_latency_cpu(model, dummy_input, n_warmup=50, n_test=200):
    batch_size = dummy_input.shape[0]
    # prévenir pytorch qu'on est pas en entrainement mais en inférence/évaluation
    model.eval()
    # mettre le modèle sur le cpu
    model = model.to('cpu')

    # warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    # real test
    timings = np.zeros((n_test, 1))
    for i in range(n_test):
        t1 = time.perf_counter()
        _ = model(dummy_input)
        t2 = time.perf_counter()
        timings[i] = t2-t1
    # time.perf_counter() returns time in s -> converted to ms (*1000)
    mean_syn = np.sum(timings) / n_test * 1000
    std_syn = np.std(timings) * 1000
    print(f'Inference time CPU (ms/image):{mean_syn/batch_size:.3f} ms +/- {std_syn/batch_size:.3f} ms')
    print(f'FPS CPU: {batch_size*1000/mean_syn}')
    return mean_syn, std_syn


@torch.no_grad()
def measure_latency_gpu(model, dummy_input, n_warmup=50, n_test=200):
    batch_size = dummy_input.shape[0]
    # prévenir pytorch qu'on est pas en entrainement mais en inférence/évaluation
    model.eval()
    # mettre le modèle sur le gpu
    model.to('cuda')
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)

    timings = np.zeros((n_test, 1))

    # gpu warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    # mesure performance
    with torch.no_grad():
        for rep in range(n_test):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # gpu sync to make sure that the timing is correct
            torch.cuda.synchronize()
            # return time in milliseconds
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / n_test
    std_syn = np.std(timings)
    print(f'Inference time GPU (ms/image): {mean_syn/batch_size:.3f} ms +/- {std_syn/batch_size:.3f} ms')
    print(f'FPS GPU: {batch_size*1000/mean_syn}')
    return mean_syn, std_syn

@torch.no_grad()
def benchmark(model, dummy_input, n_warmup=50, n_test=200, plot=False):
    batch_size = dummy_input.shape[0]
    dummy_input = dummy_input.to('cpu')
    mean_syn_cpu, std_syn_cpu = measure_latency_cpu(model, dummy_input, n_warmup, n_test)
    dummy_input = dummy_input.to('cuda')
    mean_syn_gpu, std_syn_gpu = measure_latency_gpu(model, dummy_input, n_warmup, n_test)
    # print('Sparsité:', get_model_sparsity(model))
    # get_layer_sparsity(model)
    print(f'Nombre de paramètres: {get_num_parameters(model)/1e6:.3f} M')
    print(f'Taille du modèle: {get_model_size(model)/MiB:.3f} MiB')
    print(f'Nombre de MACs: {get_model_macs(model, dummy_input)/1e6/batch_size:.3f} M')
    if plot:
        plot_num_parameters_distribution(model)
        plot_weight_distribution(model)
    return mean_syn_cpu, std_syn_cpu, mean_syn_gpu, std_syn_gpu

