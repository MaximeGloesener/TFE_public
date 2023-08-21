import torch
import datetime
import time
from tqdm import tqdm
import numpy as np
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

from trt.utils import *
from trt import common
from trt.calibrator import Calibrator
from torch.utils.data import DataLoader
from torchvision.transforms import *
from torchvision.datasets import *
from models.vgg_tiny import VGG
import os 

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

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



def quantization_main():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    onnx_model_file = 'vgg.onnx'
    inference_times = 100
    
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
    # ==> pytorch test
    state_dict = torch.load('models/pretrained_state_dicts_cifar10/vgg.cifar.pretrained.pth')
    model = VGG()
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model_torch = model
    input_torch = torch.randn(1, 3, 32, 32).cuda()
    model_torch.eval()
    model_torch.cuda()
    
    # warm up
    for _ in range(20):
        out = model_torch(input_torch)

    t_begin = time.monotonic()
    with torch.no_grad():
        for i in tqdm(range(inference_times)):
            outputs_torch = model_torch(input_torch)
    t_end = time.monotonic()
    
    torch_time = (t_end - t_begin)/inference_times

    # ==> trt test
    with build_engine_onnx(TRT_LOGGER, onnx_model_file) as engine:

        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            # Load a normalized test case into the host input page-locked buffer.
            np.copyto(inputs[0].host, input_torch.cpu().numpy().ravel())
            
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            
            t_begin = time.monotonic()
            for i in tqdm(range(inference_times)):
                trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t_end = time.monotonic()
            trt_time = (t_end - t_begin)/inference_times
            with open('modelfullp.engine', "wb") as f:
                f.write(engine.serialize()) 

    # ==> trt int8 quantization test
    calibration_cache = 'calib_cache_7'
    training_data = 'data/cifar10'
    # get the calibrator for int8 post-quantization
    calib = Calibrator(dataloader["train"], cache_file=calibration_cache)

    with build_engine_onnx_int8(TRT_LOGGER, onnx_model_file, calib) as engine:

        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            # Load a normalized test case into the host input page-locked buffer.
            np.copyto(inputs[0].host, input_torch.cpu().numpy().ravel())
            
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            t_begin = time.time()
            for i in tqdm(range(inference_times)):
                trt_int8_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t_end = time.time()
            trt_int8_time = (t_end - t_begin)/inference_times
        with open('model6.engine', "wb") as f:
            f.write(engine.serialize()) 

    print('==> Torch time: {:.5f} ms'.format(torch_time))
    print(f'Torch FPS = {1/torch_time:.2f}')
    print('==> TRT time: {:.5f} ms'.format(trt_time))
    print(f'TRT FPS = {1/trt_time:.2f}')
    print('==> TRT INT8 time: {:.5f} ms'.format(trt_int8_time))
    print(f'TRT INT8 FPS = {1/trt_int8_time:.2f}')

quantization_main()