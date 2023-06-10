import os
import pickle as pkl
from pathlib import PurePath
from typing import Dict, Optional, Tuple, Union

import numpy as np

import torch
import torchvision.transforms as T
import torch.nn.functional as F

from tqdm import tqdm


def load_maps(
    path_to_maps: Union[str, PurePath],
):
    """
    Loads the maps generated by gen_saliency_maps.

    Parameters
    ----------
    path_to_maps : Union[str, PurePath]
        The path to the maps.

    Returns
    -------
    Tuple[Dict[Union[str, PurePath], Tuple[int, np.ndarray, np.ndarray]]]
        A dictionary containing the maps.
    """

    if isinstance(path_to_maps, str):
        path_to_maps = PurePath(path_to_maps)

    maps = dict()
    pbar = tqdm(
        total=os.path.getsize(path_to_maps),
        unit="B",
        unit_scale=True,
        desc="Loading maps to memory"
    )

    with open(path_to_maps, "rb") as f:
        while True:
            try:
                idx, img, map = pkl.load(f)
                maps[int(idx)] = (np.array(img), np.array(map))
            except EOFError:
                break

            pbar.update(f.tell() - pbar.n)

    pbar.close()
    return maps


def norm_map(
    map: np.ndarray
):
    """
    Normalizes the map.

    Parameters
    ----------
    map : np.ndarray
        The map to normalize.

    Returns
    -------
    np.ndarray
        The normalized map.
    """

    return (map - np.min(map)) / (np.max(map) - np.min(map))


class DAUC():

    def __init__(
        self,
        model: torch.nn.Module,
        maps: Dict[int, Tuple[np.ndarray, np.ndarray]],
        device: torch.device,
        masking_steps: Optional[int] = None,
    ):
        self.model = model
        self.maps = maps
        self.device = device
        self.masking_steps = masking_steps

        self.keys = list(self.maps.keys())
        self.current_idx = 0

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.keys):
            raise StopIteration

        result = self._get_dauc()
        self.current_idx += 1
        return result

    def _get_masked_image(
        self,
        img: np.ndarray,
        saliency_map: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies the mask to the image.

        Parameters
        ----------
        img : np.ndarray
            The image with three dimensions in channel first order
            (channels, height, width).
        saliency_map : np.ndarray
            The saliency map with two dimensions, height and width.
            If the saliency map has three dimensions, the sum over
            the first dimension is calculated. The saliency map
            is then normalized.
        mask : Optional[np.ndarray], optional
            The mask to apply to the image, by default None.
            Can be used to apply a mask from a previous masking step.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The masked image and the mask.
        """

        if len(saliency_map.shape) == 3:
            saliency_map = norm_map(np.sum(saliency_map, axis=0)).squeeze()
            saliency_map = saliency_map * 255
            saliency_map = saliency_map.astype(np.uint8)

        r = img.shape[1]/saliency_map.shape[0]
        assert r == img.shape[2]/saliency_map.shape[1], \
            "The image and the saliency map must have the same aspect ratio."

        mask = np.ones((img.shape[1], img.shape[2])) if mask is None else mask
        steps = np.count_nonzero(mask == 0)
        steps = int((steps // r**2))

        # get the indices of the sorted saliency map in descending order
        sort_idx = np.unravel_index(
            np.argsort(saliency_map.ravel(), axis=None),
            saliency_map.shape
        )

        for i, j in zip(sort_idx[0][::-1][steps:], sort_idx[1][::-1][steps:]):
            if saliency_map[i, j] == 0:
                mask *= 0
                return img * mask[np.newaxis, :, :], mask

            mask[int(i*r):int((i+1)*r), int(j*r):int((j+1)*r)] = 0
            return img * mask[np.newaxis, :, :], mask

    def _get_dauc(self):
        img, map = self.maps[self.keys[self.current_idx]]
        masked_image = img
        mask = np.ones((img.shape[1], img.shape[2]))
        outputs = []
        proportions = []

        pbar = tqdm(desc=f'Masking image {self.current_idx}')

        i = 0
        masked_imgs = []
        while mask.sum() > 0:

            proportions.append(np.count_nonzero(mask==0) / mask.size)

            masked_image, mask = self._get_masked_image(
                img=img,
                saliency_map=map,
                mask=mask,
            )
            norm_img = self.transform(np.moveaxis(masked_image, 0, -1))
            norm_img = norm_img.type(torch.FloatTensor).unsqueeze(0)
            masked_imgs.append(norm_img)

            if len(masked_imgs) >= 16 or mask.sum() == 0:
                masked_imgs = torch.cat(masked_imgs, dim=0)
                prediction = self.model(masked_imgs.to(self.device))
                outputs.extend(prediction.detach().cpu().tolist())
                masked_imgs = []

            pbar.update(1)
            i += 1

            if self.masking_steps is not None and i >= self.masking_steps:
                break

        outputs = np.array(outputs)
        proportions = np.array(proportions)
        outputs = F.softmax(torch.tensor(outputs), dim=1).numpy()

        inital_prediction = np.argmax(outputs[0])
        deletion_score = outputs[:, inital_prediction]
        deletion_score /= deletion_score.max()

        return (
            np.trapz(deletion_score, proportions),
            deletion_score,
            proportions
        )


class IAUC():

    def __init__(
        self,
        model: torch.nn.Module,
        maps: Dict[int, Tuple[np.ndarray, np.ndarray]],
        device: torch.device,
        masking_steps: Optional[int] = None,
    ):
        self.model = model
        self.maps = maps
        self.device = device
        self.masking_steps = masking_steps

        self.keys = list(self.maps.keys())
        self.current_idx = 0

        self.blur = T.Compose([
            T.ToTensor(),
            T.GaussianBlur(
                kernel_size=5,
            ),
            T.GaussianBlur(
                kernel_size=9,
            ),
            T.GaussianBlur(
                kernel_size=17,
            ),
        ])

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.keys):
            raise StopIteration

        result = self._get_iauc()
        self.current_idx += 1
        return result

    def _get_masked_image(
        self,
        img: np.ndarray,
        saliency_map: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies the mask to the image.

        Parameters
        ----------
        img : np.ndarray
            The image with three dimensions in channel first order
            (channels, height, width).
        saliency_map : np.ndarray
            The saliency map with two dimensions, height and width.
            If the saliency map has three dimensions, the sum over
            the first dimension is calculated. The saliency map
            is then normalized.
        mask : Optional[np.ndarray], optional
            The mask to apply to the image, by default None.
            Can be used to apply a mask from a previous masking step.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The masked image and the mask.
        """

        if len(saliency_map.shape) == 3:
            saliency_map = norm_map(np.sum(saliency_map, axis=0)).squeeze()
            saliency_map = saliency_map * 255
            saliency_map = saliency_map.astype(np.uint8)

        r = img.shape[1]/saliency_map.shape[0]
        assert r == img.shape[2]/saliency_map.shape[1], \
            "The image and the saliency map must have the same aspect ratio."

        mask = np.zeros((img.shape[1], img.shape[2])) if mask is None else mask
        steps = np.count_nonzero(mask == 1)
        steps = int((steps // r**2))

        # get the indices of the sorted saliency map in descending order
        sort_idx = np.unravel_index(
            np.argsort(saliency_map.ravel(), axis=None),
            saliency_map.shape
        )
        blured_img = self.blur(np.moveaxis(img, 0, -1)).numpy()

        for i, j in zip(sort_idx[0][::-1][steps:], sort_idx[1][::-1][steps:]):
            if saliency_map[i, j] == 0:
                mask = np.ones((mask.shape[0], mask.shape[1]))
            else:
                mask[int(i*r):int((i+1)*r), int(j*r):int((j+1)*r)] = 1

            masked_img = img * mask[np.newaxis, :, :]
            masked_img += blured_img * (1 - mask[np.newaxis, :, :])

            return masked_img, mask

    def _get_iauc(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculates the integrated IAUC for the current image.

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray]
            The IAUC, the insertion scores and the proportions.
        """
        img, map = self.maps[self.keys[self.current_idx]]
        masked_image = img
        mask = np.zeros((img.shape[1], img.shape[2]))
        outputs = []
        proportions = []

        pbar = tqdm(desc=f'Masking image {self.current_idx}')

        i = 0
        masked_imgs = []
        while mask.sum() < mask.size:

            proportions.append(np.count_nonzero(mask==1) / mask.size)

            masked_image, mask = self._get_masked_image(
                img=img,
                saliency_map=map,
                mask=mask,
            )
            norm_img = self.transform(np.moveaxis(masked_image, 0, -1))
            norm_img = norm_img.type(torch.FloatTensor).unsqueeze(0)
            masked_imgs.append(norm_img)

            if len(masked_imgs) >= 16 or mask.sum() <= mask.size:
                masked_imgs = torch.cat(masked_imgs, dim=0)
                prediction = self.model(masked_imgs.to(self.device))
                outputs.extend(prediction.detach().cpu().tolist())
                masked_imgs = []

            pbar.update(1)
            i += 1

            if self.masking_steps is not None and i >= self.masking_steps:
                break

        outputs = np.array(outputs)
        proportions = np.array(proportions)
        outputs = F.softmax(torch.tensor(outputs), dim=1).numpy()

        inital_prediction = np.argmax(
            self.model(
                self.transform(
                    np.moveaxis(img, 0, -1)
                ).unsqueeze(0).to(self.device)
            ).detach().cpu().numpy()
        )

        insertion_score = outputs[:, inital_prediction]
        insertion_score /= insertion_score.max()

        return (
            np.trapz(insertion_score, proportions),
            insertion_score,
            proportions
        )
