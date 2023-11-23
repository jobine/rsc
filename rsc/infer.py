import glob
import os
import numpy as np
import torch
from PIL import Image

from rsc import utils
from rsc.environment import RESNET101_MODEL_PATH, MODEL_DIR, WTCR_DATA_DIR

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # model = utils.get_model(os.path.join(MODEL_DIR, 'resnet101-1700721031.pt'), ['Cloudy', 'Sunny', 'Rainy'], device, vis_model=True)
    model = torch.load(os.path.join(MODEL_DIR, 'resnet101-1700720669.pth'), map_location=device)
    model.eval()

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