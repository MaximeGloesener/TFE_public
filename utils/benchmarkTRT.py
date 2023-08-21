import torch
from torch2trt import torch2trt
import torch
import numpy as np 
import torch.nn.functional as F
import os 
from utils.benchmark import *
from models.vgg_tiny import VGG
import numpy as np
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
from torch2trt import TRTModule 




example_input = torch.rand((64,3,32,32)).cuda()

"""for model_name in os.listdir('results_quant224/'):
    print(model_name)
    model_trt = TRTModule()

    model_trt.load_state_dict(torch.load(f"results_quant224/{model_name}", map_location="cpu"))
    measure_latency_gpu(model_trt, example_input.cuda(), n_warmup=80, n_test=500)

"""



model_trt = TRTModule()
model_trt.load_state_dict(torch.load(f"results_quant224/abtestt.pth", map_location="cpu"))

measure_latency_gpu(model_trt, example_input.cuda(), n_warmup=80, n_test=500)

