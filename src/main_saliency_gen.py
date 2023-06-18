import os
from pathlib import PurePath

import torch

from models import model_loader
from utils import dataset_loader, saliency_gen
from engine import train

WEIGHTS_PATH = "./output/weights/resnet50_9.pth"
DATASET_NAME = 'CUB_200_2011'
MODEL_NAME = 'resnet50'
OUT_DIR = './output'


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_path = PurePath(WEIGHTS_PATH)

    dataset = dataset_loader.load_dataset(
        dataset_name=DATASET_NAME,
        resize=(128, 128)
    )
    model, tuned = model_loader.load_cnn(
        model_name=MODEL_NAME,
        pretrained=True,
        out_features=200,
        load_weights=weights_path,
    )

    batch_size = dataset_loader.get_best_batch_size(
        model=model,
        dataset=dataset,
        device=device
    )
    print(f"Batch size: {batch_size}")
    if not tuned:
        for epoch in range(10):
            stats, stop = train.train_one_epoch(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
            )

            os.makedirs(weights_path.parent, exist_ok=True)
            file = weights_path.parent / f"{MODEL_NAME}_{epoch}.pth"
            torch.save(model.state_dict(), open(file, 'wb'))

    saliency_gen.gen_saliency_maps(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=OUT_DIR
    )
