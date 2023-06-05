
import os
import pickle as pkl
from collections import defaultdict
from operator import attrgetter
from pathlib import PurePath
from typing import Dict, List, Optional, Tuple, Union
# from threading import Thread

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import datasets
from captum.attr import Saliency, IntegratedGradients, LayerGradCam
from captum.attr import visualization as viz
from pytorch_grad_cam import ScoreCAM

from tqdm import tqdm


def _viz_worker(
    maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
):
    """
    Visualises the saliency maps and saves them to disk.

    Parameters
    ----------
    maps : Dict[str, Tuple[np.ndarray, np.ndarray]]
        A dictionary mapping a string filepath containing the attribution to a
        tuple of the original image and the attribution map.
    """
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


def save_attr(
    image: np.ndarray,
    attr: np.ndarray,
    path: Union[str, PurePath],
    filename: str,
    methods: Optional[List[str]] = ["original_image", "heat_map", "blended_heat_map"],
    signs: Optional[List[str]] = ["all", "positive", "positive"],
):
    """
    Saves the attribution map to disk.
    If methods and signs are specified, they need to have the same length.

    Parameters
    ----------
    image : np.ndarray
        The original image.
    attr : np.ndarray
        The attribution map.
    path : Union[str, PurePath]
        The path to save the attribution map to.
    filename : str
        The filename of the attribution map.
    methods : Optional[List[str]], optional
        Visualization methods for the plot, by default ["original_image", "heat_map", "blended_heat_map"]
    signs : Optional[List[str]], optional
        What attibutes to choose for each method, by default ["all", "positive", "positive"]
    """
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
    """
    Generates saliency maps for the given model and dataset.

    Parameters
    ----------
    model : nn.Module
        The model to generate saliency maps for.
    dataset : datasets.VisionDataset
        The dataset to generate saliency maps for.
    device : torch.device
        The device to run the model on.
    output_dir : Union[str, PurePath]
        The directory to save the saliency maps to.
    gradcamlayers : Optional[List[str]], optional
        A list of layers for GradCAM, by default all 'conv1' layers.

    Raises
    ------
    ValueError
        If gradcamlayers contains a layer that is not in the model.
    """

    if isinstance(output_dir, str):
        output_dir = PurePath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

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

            # If the maps should be plotted or not,
            # please comment in or out the following lines
            # (VERY TIME CONSUMING!):
            # thread = Thread(
            #     target=_viz_worker,
            #     args=(maps, )
            # )
            # workers.append(thread)
            # thread.start()
            maps = dict()

    for thread in workers:
        thread.join()
