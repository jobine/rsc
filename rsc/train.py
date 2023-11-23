import time

import cv2
import numpy as np
import pandas as pd
import os
import json
import random
import glob
import torch
import torchvision
from PIL import Image
from albumentations import ShiftScaleRotate, Compose, Resize, OneOf, GaussianBlur, GaussNoise, Normalize
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet101_Weights
from tqdm import tqdm

from rsc import utils
from rsc.environment import *


class CustomDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        image = np.array(Image.open(os.path.join(WTCR_DATA_DIR, 'train_dataset', self.image_files[i])).convert('RGB'))
        labels = self.labels[i]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, labels


def load_annotation(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return pd.json_normalize(data['annotations'])


def load_data(path, transforms, random_state=0):
    annotation_path = os.path.join(path, 'train_dataset', 'train.json')
    train_ds = load_annotation(annotation_path)

    weather_encoder = LabelEncoder().fit(train_ds['weather'])

    train_ds['weather'] = weather_encoder.transform(train_ds['weather'])
    train_ds['filename'] = train_ds['filename'].str.replace('\\', os.sep, regex=True)

    x_train, x_test, y_train, y_test = train_test_split(train_ds['filename'].values, train_ds['weather'].values, random_state=random_state, shuffle=True, stratify=train_ds['weather'])



    sets = {
        'train': (x_train, y_train), 'test': (x_test, y_test)
    }
    datasets = {
        x: CustomDataset(sets[x][0], sets[x][1], transforms[x]) for x in sets.keys()
    }
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=8, num_workers=4, pin_memory=True) for x in sets.keys()
    }

    return dataloaders


def train(i, model, dataloader, optimizer, loss_func, device):
    overall_loss = 0.

    with tqdm(dataloader, total=len(dataloader), desc=f'Training, phase {i} :') as loader:
        for data, weather in loader:
            data, weather = data.to(device), weather.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = loss_func(output, weather)

            overall_loss += loss.item()

            loader.set_postfix(loss=overall_loss / len(dataloader))

            loss.backward()
            optimizer.step()


def validate(i, model, dataloader, device):
    model.eval()

    weather_acc = 0.

    with torch.no_grad():
        with tqdm(dataloader, total=len(dataloader), desc=f'Validation, phase {i} :') as loader:
            for data, weather in loader:
                data, weather = data.to(device), weather.to(device)

                output = model(data)

                # weather_acc += accuracy_score(weather.cpu().numpy(), torch.argmax(output, dim=1).cpu().numpy())
                weather_acc += accuracy_score(weather.cpu().detach().numpy(), torch.argmax(output, dim=1).cpu().detach().numpy())

                loader.set_postfix(accuracy_weather=weather_acc / len(dataloader))

    model.train()


if __name__ == '__main__':
    random_state = 42
    utils.set_seed(random_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    model = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)

    model = model.to(device)

    lr = 1e-4
    epochs = 5

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_func = torch.nn.CrossEntropyLoss().to(device)

    transforms = {
        x: Compose([
            Resize(224, 224),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_REPLICATE),
            OneOf([
                GaussianBlur(),
                GaussNoise(),
            ], p=0.2),
            Normalize(),
            ToTensorV2()
        ]) if x == 'train' else Compose([
            Resize(224, 224),
            Normalize(),
            ToTensorV2()
        ]) for x in ['train', 'test']
    }

    dataloaders = load_data(WTCR_DATA_DIR, transforms, random_state=random_state)

    utils.empty_cache()

    for epoch in range(1, epochs + 1):
        train(epoch, model, dataloaders['train'], optimizer, loss_func, device)
        validate(epoch, model, dataloaders['test'], device)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'resnet101-{int(time.time())}.pt'))

    # test
    model.eval()
    train_ds = load_annotation(os.path.join(WTCR_DATA_DIR, 'train_dataset', 'train.json'))
    weather_encoder = LabelEncoder().fit(train_ds['weather'])

    images = glob.glob(os.path.join(WTCR_DATA_DIR, 'test_dataset', 'test_images', '*'))

    for _ in range(5):
        image = np.array(Image.open(np.random.choice(images)).convert('RGB'))
        print(image.shape)

        tensor = transforms['test'](image=image)['image'].unsqueeze(0).to(device)
        print(tensor.shape)
        output = model(tensor)

        output = torch.argmax(output, dim=1).cpu().detach().numpy()
        print(output)
        plt.imshow(image)
        plt.title(f'{weather_encoder.inverse_transform(output)[0]}')
        plt.axis('off')
        plt.show()