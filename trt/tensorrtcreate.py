import tensorrt as trt
from trt import common 
from trt.utils import *
import torch
import numpy as np
import time
from tqdm import tqdm
from trt.calibrator import Calibrator
from torch.utils.data import DataLoader
from torchvision.transforms import *
from torchvision.datasets import *


image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        #Normalize( **NORMALIZE_DICT['cifar10']),
    ]),
    "test": Compose([
        ToTensor(),
        #Normalize( **NORMALIZE_DICT['cifar10']),
    ]),
}

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

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
onnx_file = 'vgg.onnx'
calibration_cache = 'calib_cache'
# get the calibrator for int8 post-quantization
calib = Calibrator(dataloader["train"], cache_file=calibration_cache)
with build_engine_onnx_int8(TRT_LOGGER, onnx_file, calib) as engine:
    with open('vggquanttest.engine', "wb") as f:
        f.write(engine.serialize()) 
