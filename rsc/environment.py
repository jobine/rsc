import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../model')
# RESNET18_MODEL_PATH = os.path.join(MODEL_DIR, 'resnet18-f37072fd.pth')
RESNET18_MODEL_PATH = os.path.join(MODEL_DIR, 'resnet101-1700494450.pth')
RESNET101_MODEL_PATH = os.path.join(MODEL_DIR, 'resnet101-cd907fc2.pth')
HYMENOPTERA_DATA_DIR = os.path.join(BASE_DIR, '../data', 'hymenoptera')
RSC_DATA_DIR = os.path.join(BASE_DIR, '../data', 'rsc')
WTCR_DATA_DIR = os.path.join(BASE_DIR, '../data', 'WTCR')