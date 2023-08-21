# import pretrained models 
from models.repvgg import repvgg_a0, repvgg_a1, repvgg_a2
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.resnet import resnet20, resnet32, resnet44, resnet56 

import torch 

def get_model(model_name, dataset_name):
    # get num_classes
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100

    # load le checkoint du modèle pré-entrainé
    if dataset_name == 'cifar10':
        checkpoint = torch.load(f'models/pretrained_state_dicts_cifar10/cifar10_{model_name}.pt', map_location='cpu')
    elif dataset_name == 'cifar100':
        checkpoint = torch.load(f'models/pretrained_state_dicts_cifar100/cifar100_{model_name}.pt', map_location='cpu')

    # load le modèle (on appelle la fonction qui a le même nom que le modèle)
    model_fn = globals()[model_name]  
    model = model_fn(num_classes).cuda()
    # load le checkpoint 
    model.load_state_dict(checkpoint)
    return model