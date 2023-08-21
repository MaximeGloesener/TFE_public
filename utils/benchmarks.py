# script for benchmarking model speed gpu/cpu/macs/num params/size
import os 
import pandas as pd 
import torch 
from utils.benchmark import benchmark, get_num_parameters, get_model_size, get_model_macs

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

import os
import torch
import pandas as pd

# num params for base model to get exact compression ratio
base_models = {'repvgg': 7840874, 'resnet56': 855770, 'vgg16': 15253578}

df_list = []  # List to store dataframes for each iteration

for result in os.listdir('results/'):
    if result.endswith('.pth'):
        model = torch.load('results/' + result)
        model.eval()
        batch_size = 256
        example_input = torch.rand((batch_size, 3, 32, 32))
        model_name = result.split('_')[0]
        
        with torch.inference_mode():
            mean_syn_cpu, std_syn_cpu, mean_syn_gpu, std_syn_gpu = benchmark(model, example_input)
            num_params = f'{get_num_parameters(model) / 1e6:.3f}'
            model_size = f'{get_model_size(model) / MiB:.3f}'
            model_macs = f'{get_model_macs(model, example_input.cuda()) / 1e6 / batch_size:.3f}'
            compression_ratio = f'{base_models[model_name] / get_num_parameters(model):.3f}'
            fps_cpu = batch_size*1000/mean_syn_cpu
            fps_gpu = batch_size*1000/mean_syn_gpu
        # Create a dataframe for the current iteration
        df = pd.DataFrame({'model': [model_name], 'CPU time (ms)': [mean_syn_cpu], 'std_cpu': [std_syn_cpu],
                        'GPU time (ms)': [mean_syn_gpu], 'std_gpu': [std_syn_gpu],
                        'num_params (M)': [num_params], 'model_size (MiB)': [model_size],
                        'model_macs (M)': [model_macs], 'compression_ratio': [compression_ratio],
                        'batch_size': [batch_size], 'fps_cpu': [fps_cpu], 'fps_gpu': [fps_gpu]})
        df_list.append(df)  # Append the dataframe to the list

# Concatenate all dataframes in the list vertically
final_df = pd.concat(df_list, ignore_index=True)

# Save the final dataframe to a CSV file
final_df.to_csv(f'results/benchmark{batch_size}.csv', index=False)
