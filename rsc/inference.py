import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights

from environment import HYMENOPTERA_DATA_DIR, RESNET18_MODEL_PATH, RESNET101_MODEL_PATH, MODEL_DIR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_transform(img_rgb, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    img = transform(img_rgb)
    return img


def get_image_names(img_dir, format='jpeg'):
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

    # checkpoints = torch.load(model_path)
    # model.load_state_dict(checkpoints)

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device=device)

    model = model.to(device)
    model.eval()

    return model


def infer_images(model, img_dir, classes, device, vis_model=False):
    time_total = 0
    img_list, img_pred = list(), list()
    vis_row = 4
    img_names = get_image_names(img_dir)
    num_img = len(img_names)

    with torch.no_grad():
        for idx, img_name in enumerate(img_names):
            img_path = os.path.join(img_dir, img_name)

            # step 1/4: path->img
            img_rgb = Image.open(img_path).convert('RGB')

            # step 2/4: img->tensor
            img_tensor = image_transform(img_rgb).unsqueeze(0)

            # step 3/4: tensor->vector
            time_tick = time.time()
            outputs = model(img_tensor.to(device))
            time_tock = time.time()

            # step 4/4: visualization
            s, predicted = torch.max(outputs, 1)
            pred_str = classes[int(predicted.item())]

            if vis_model:
                img_list.append(img_rgb)
                img_pred.append(pred_str)

                if (idx + 1) % ((vis_row - 1) * vis_row) == 0 or num_img == (idx + 1):
                    for i in range(len(img_list)):
                        plt.subplot(vis_row - 1, vis_row, i + 1).imshow(img_list[i])
                        plt.title(f'{img_names[i]}:{img_pred[i]}')

                    plt.show()
                    plt.close()
                    img_list, img_pred = list(), list()

            time_s = time_tock - time_tick
            time_total += time_s

            print(f'{idx + 1:0{len(str(num_img))}}/{num_img}: {img_name} {time_s:.3f}s\n')

    print(f'device:{device} total time:{time_total:.1f}s mean:{time_total/num_img:.3f}s')
    # torch.save(model, os.path.join(MODEL_DIR, f'resnet101-{int(time.time())}.pth'))


if __name__ == '__main__':
    classes = ['ants', 'bees']
    vis_model = True

    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = get_model(RESNET101_MODEL_PATH, classes, device, vis_model=vis_model)
    infer_images(model, HYMENOPTERA_DATA_DIR, classes, device, vis_model=vis_model)
