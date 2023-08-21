# quant using pytorch tensorrt api 
import torch
import numpy as np 
import torch.nn.functional as F
from utils.benchmark import *
import torch
from models.vgg_tiny import VGG
import numpy as np
import torch
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
from models.resnet import resnet56


checkpoint = torch.load("models/pretrained_state_dicts_cifar10/cifar10_resnet56.pt", map_location="cpu")
model = resnet56(10)
model.load_state_dict(checkpoint)
model.eval()
device = torch.device('cuda')
model.to(device)
example_input = torch.rand((1,3,32,32))

print('modèle initial')
measure_latency_gpu(model, example_input.cuda())



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

# Datas
NORMALIZE_DICT = {
   'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    }
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
        batch_size=1,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True,
    )


print('modèle initial')

import torch_tensorrt

calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(dataloader["train"],
                                              use_cache=False,
                                              algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
                                              device=torch.device('cuda:0'))


compile_spec = {
         "inputs": [torch_tensorrt.Input([1, 3, 32, 32])],
         "enabled_precisions": torch.int8,
         "calibrator": calibrator,
         "truncate_long_and_double": True     
     }
trt_ptq = torch_tensorrt.compile(model, **compile_spec)

print(trt_ptq)
print(type(trt_ptq))

"""dummy_input = torch.rand((1,3,32,32)).cuda()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

timings = np.zeros((100, 1))


# gpu warmup
for _ in range(20):
    _ = trt_ptq(dummy_input)
# mesure performance
with torch.no_grad():
    for rep in range(100):
        starter.record()
        _ = trt_ptq(dummy_input)
        ender.record()
        # gpu sync to make sure that the timing is correct
        torch.cuda.synchronize()
        # return time in milliseconds
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / 100
std_syn = np.std(timings)
print('modèle optimisé')
print(f'Inference time GPU (ms/image): {mean_syn/1:.3f} ms +/- {std_syn/1:.3f} ms')
print(f'FPS GPU: {1*1000/mean_syn}')"""

acc, loss = evaluate(trt_ptq, dataloader["test"])
print(acc, loss)




"""# save 
torch.jit.save(trt_ptq, "trt_ts_module.ts")

# inference
trt_ts_module = torch.jit.load("trt_ts_module.ts")
result = trt_ts_module(example_input.cuda())
print('modèle final')
acc, loss = evaluate(trt_ts_module, dataloader["test"])
print(f"Accuracy: {acc:.2f}%, Loss: {loss:.4f}")"""