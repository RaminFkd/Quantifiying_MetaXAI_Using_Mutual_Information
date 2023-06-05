
import torch
import torchvision.transforms as T

from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from utils import metrics
from models import model_loader


WEIGHTS_PATH = './output/weights/ResNet50.pth'
PATH_TO_MAPS = './output/saliency/saliency_maps.pkl'


if __name__=='__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}.')

    print('Loading model...')
    model, _ = model_loader.load_cnn(
        model_name='resnet50',
        pretrained=True,
        out_features=200,
        load_weights=WEIGHTS_PATH,
    )
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    maps = metrics.load_maps(path_to_maps=PATH_TO_MAPS)
    dauc = metrics.DAUC(
        model=model,
        maps=maps,
        device=device,
    )
    d_auc, _, _ = next(dauc)
    print(d_auc)
