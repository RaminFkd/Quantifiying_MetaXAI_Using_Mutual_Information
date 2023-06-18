import os
import pickle as pkl
from operator import attrgetter
from typing import Optional, List, Union, Dict, Tuple
from pathlib import PurePath
from collections import defaultdict

from threading import Thread

from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from captum.attr import Saliency, IntegratedGradients, LayerGradCam
from captum.attr import visualization as viz
from pytorch_grad_cam import ScoreCAM

import models.ModelLoader
import engine.ModelTrain

import utils.DatasetLoader
from utils.DatasetLoader import get_best_batch_size

def _viz_worker(
    maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
):
    while len(maps) > 0:
        key, val = maps.popitem()
        image, attr = val

        if "gradcam" in key.parts:
            save_attr(
                np.transpose(image, (1, 2, 0)),
                np.transpose(attr, (1, 2, 0)),
                key.parent,
                key.name,
                methods = ["original_image", "heat_map"],
                signs = ["all", "positive"]
            )
        else:
            save_attr(
                np.transpose(image, (1, 2, 0)),
                np.transpose(attr, (1, 2, 0)),
                key.parent,
                key.name,
            )


def _recursive_getattr(obj, attr_list):
    if len(attr_list) == 1:
        return getattr(obj, attr_list[0])
    else:
        return _recursive_getattr(getattr(obj, attr_list[0]), attr_list[1:])


def get_class_accuracy(
    model: nn.Module,
    dataset: datasets.VisionDataset,
    transform: nn.Module,
    device: torch.device,
) -> defaultdict:

    batch_size = get_best_batch_size(
        model=model,
        dataset=dataset,
        device=device
    ) // 2

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=8,
    )

    # Get class accuracy
    model.eval()
    model.to(device)
    class_correct = defaultdict(list)
    for _, batch in tqdm(dataloader):
        images, labels = batch
        images = transform(images)
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        outputs = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        for label, pred in zip(labels, predicted):
            class_correct[label.item()].append((label == pred).item())

    for label, correct in class_correct.items():
        class_correct[label] = sum(correct)/len(correct)

    print(class_correct)


def save_attr(
    image: np.ndarray,
    attr: np.ndarray,
    path: Union[str, PurePath],
    filename: str,
    methods: Optional[List[str]] = ["original_image", "heat_map", "blended_heat_map"],
    signs: Optional[List[str]] = ["all", "positive", "positive"],
):
    assert len(methods) == len(signs), \
        "methods and signs must have the same length"

    if isinstance(path, str):
        path = PurePath(path)

    try:
        fig, _ = viz.visualize_image_attr_multiple(
            attr,
            image,
            methods=methods,
            signs=signs,
            show_colorbar=True,
            use_pyplot=False,
        )
        fig.savefig(path / filename)
    except AssertionError:
        pass


def gen_saliency_maps(
    model: nn.Module,
    dataset: datasets.VisionDataset,
    device: torch.device,
    output_dir: Union[str, PurePath],
    gradcamlayers: Optional[List[str]] = None,
    **dataloader_kwargs
):

    if isinstance(output_dir, str):
        output_dir = PurePath(output_dir)

    model_layers = [name for name, _ in model.named_modules()]

    if gradcamlayers is None:
        gradcamlayers = [name for name in model_layers if 'conv1' in name]
    else:
        diff = set(gradcamlayers) - set(model_layers)
        if len(diff) > 0:
            valid_layer_names = ", ".join(model_layers)
            raise ValueError(
                f"""Layer(s) {str(diff)} not found in model,
                valid layer names are: {valid_layer_names}"""
            )

    # Define the dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=8,
        **dataloader_kwargs
    )

    # Define the attribution methods
    saliency = Saliency(model)
    ig = IntegratedGradients(model)

    keys = [attrgetter(name) for name in gradcamlayers]
    score_cam = ScoreCAM(model, target_layers=[keys[0](model)], use_cuda=True)
    gradcam_dict = {
        name: LayerGradCam(model, key(model)) for key, name in zip(keys, gradcamlayers)
    }

    norm = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    model.eval()
    model.to(device)
    counter = defaultdict(int)

    workers = []
    maps = dict()
    for index, batch in tqdm(dataloader):

        # Get the images and labels
        image, label = batch
        image = image.to(device)
        label = label.to(device)
        norm_image = norm(image)

        norm_image.requires_grad = True
        counter[label.item()] += 1

        # Get the scorecam map
        score_cam_map = score_cam(norm_image)
        key = output_dir / PurePath(
            f"score_cam/score_cam_{str(label.item()).zfill(4)}_" +
            f"{str(counter[label.item()]).zfill(4)}.png"
        )
        maps[key] = (
            index,
            image.cpu().detach().numpy().squeeze(),
            score_cam_map
        )

        # Get the saliency maps
        saliency_map = saliency.attribute(norm_image, target=label)
        key = output_dir / PurePath(
            f"saliency/saliency_{str(label.item()).zfill(4)}_" +
            f"{str(counter[label.item()]).zfill(4)}.png"
        )
        maps[key] = (
            index,
            image.cpu().detach().numpy().squeeze(),
            saliency_map.cpu().detach().numpy().squeeze()
        )

        # Get the integrated gradients
        integrated_grad = ig.attribute(norm_image, target=label)
        key = output_dir / PurePath(
            f"integrated_grad/integrated_grad_{str(label.item()).zfill(4)}_" +
            f"{str(counter[label.item()]).zfill(4)}.png"
        )
        maps[key] = (
            index,
            image.cpu().detach().numpy().squeeze(),
            integrated_grad.cpu().detach().numpy().squeeze()
        )

        # Get the gradcam maps
        gradcam_maps = {
            name: gradcam.attribute(norm_image, target=label)
            for name, gradcam in gradcam_dict.items()
        }
        for name, gradcam_map in gradcam_maps.items():
            key = output_dir / PurePath(
                f"gradcam/{name}/gradcam_{str(label.item()).zfill(4)}_" +
                f"{str(counter[label.item()]).zfill(4)}.png"
            )
            maps[key] = (
                index,
                image.cpu().detach().numpy().squeeze(),
                gradcam_map.cpu().detach().numpy().squeeze(0)
            )

        if index % 200 == 0 or index == len(dataloader) - 1:
            for key, val in maps.items():
                os.makedirs(key.parent, exist_ok=True)

                file = key.parent / f"{key.parent.name}_maps.pkl"
                with open(file, "ab") as f:
                    pkl.dump(val, f)

            # thread = Thread(
            #     target=_viz_worker,
            #     args=(maps, )
            # )
            # workers.append(thread)
            # thread.start()
            maps = dict()

    for thread in workers:
        thread.join()


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    weights_path = PurePath(
        f"./output/weights/ResNet50.pth"
    )

    dataset = utils.DatasetLoader.load_dataset(
        dataset_name='CUB_200_2011',
        resize=(224, 224)
    )
    model, tuned = models.ModelLoader.load_cnn(
        model_name='resnet50',
        pretrained=True,
        out_features=200,
        load_weights=weights_path,
    )

    batch_size = utils.DatasetLoader.get_best_batch_size(
        model=model,
        dataset=dataset,
        device=device
    )
    print(f"Batch size: {batch_size}")
    if not tuned:
        for epoch in range(10):
            stats, stop = engine.ModelTrain.train_one_epoch(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
            )

            if stop:
                os.makedirs(weights_path.parent, exist_ok=True)
                torch.save(model.state_dict(), open(weights_path, 'wb'))
                break

    gen_saliency_maps(
        model=model,
        dataset=dataset,
        device=device,
        output_dir='./output'
    )
