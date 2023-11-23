import os
import random

import torch

import numpy as np
from torch import nn
from torchvision import models
from torchvision.models import ResNet101_Weights


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def empty_cache():
    torch.cuda.empty_cache()
    torch.mps.empty_cache()


def get_image_names(img_dir, format='jpg'):
    file_names = os.listdir(img_dir)
    img_names = list(filter(lambda x: x.endswith(format), file_names))

    if len(img_names) < 1:
        raise ValueError(f'No {format} image found in the directory: {img_dir}')

    return img_names


def get_model(model_path, classes, device, vis_model=False):
    # model = torch.hub.load('pytorch/vision', 'resnet101', weights=ResNet101_Weights.DEFAULT)
    model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
    # model = models.resnet101(pretrained=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))

    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints, strict=False)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device=device)

    model = model.to(device)
    model.eval()

    return model