## Modèles 

Ce dossier reprend l'implémentation de différents modèles:

- ResNet
- VGG
- RepVGG 
- MobileNetv2

-> Les modèles sont adaptés de [pytorch_cifar_models](https://github.com/chenyaofo/pytorch-cifar-models/tree/master/pytorch_cifar_models) pour CIFAR10 et CIFAR100. 

### State dicts
-> Les state dicts des modèles sont disponibles sur un drive au lien suivant: [state dicts](https://drive.google.com/drive/folders/1Nbf7d6vb1c92gn91mKP_fuHIphtPGCXv?usp=sharing).


### Lancer un modèle
Pour lancer un modèle, il suffit de télécharger les state dicts dans un dossier (ici appelé models/state_dicts) et ensuite, de charger le modèle en suivant:

```python

import torch 
from resnet import resnet56

checkpoint = torch.load('models/state_dicts/cifar10_resnet56.pt', map_location=torch.device('cpu'))
model = resnet56(10)
model.load_state_dict(checkpoint)


```
Avec 10 qui est le nombre de classes pour le jeu de données CIFAR10, initialiser avec resnet56(100) pour le jeu de données CIFAR100. 

Les modèles servent juste d'exemple mais vous pouvez bien évidemment charger votre propre modèle. 