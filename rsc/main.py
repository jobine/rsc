import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def preprocess():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform


def load_data(transform):
    train_dataset = ImageFolder(root='data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    test_dataset = ImageFolder(root='data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    return train_loader, test_loader


if __name__ == '__main__':
    print('Hello, World!')