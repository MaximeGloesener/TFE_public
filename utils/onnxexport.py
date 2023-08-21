import numpy as np 
import random 
import torch 
import torch.onnx 
from torch.onnx import TrainingMode


# Fixer le seed pour la reproductibilité
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# model initial 
from models.vgg_tiny import VGG
from models.resnet import resnet56
checkpoint = torch.load('models/pretrained_state_dicts_cifar10v2/state_dicts/vgg.cifar.pretrained.pth', map_location="cpu")
model = VGG()
model.load_state_dict(checkpoint["state_dict"])



example_input = torch.randn(1, 3, 32, 32)

# Exporter le modèle vers ONNX
# en mode entrainement pour avoir le graphe de computation complet sans batch folding etc
# torch.onnx.export(model, example_input, "vgg.onnx", verbose=True, training=TrainingMode.TRAINING, input_names=['input'], output_names=['output'])
# en mode normal/inférence avec les optimisations
torch.onnx.export(model, example_input, "vgg2.onnx", verbose=False, input_names=['input'], output_names=['output'], export_params=True)
