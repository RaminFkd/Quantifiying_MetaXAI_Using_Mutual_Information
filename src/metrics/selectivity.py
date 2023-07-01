from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import quantus
import torch

from .metric_base import SUPPORTED_METHODS, MetricBase


class Selectivity(MetricBase):

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        normalize: bool = True,
        resize: Optional[Tuple[int, int]] = None,
        out_path: Optional[Path] = None,
        patch_size: int = 16,
        perturb_baseline: str = "black"
    ) -> None:

        super().__init__(
            model,
            device,
            normalize,
            resize,
            out_path)

        self.selectivity = quantus.Selectivity(
            patch_size=patch_size,
            perturb_baseline=perturb_baseline
        )

    def __call__(
        self,
        image: torch.Tensor,
        label: int,
        method: SUPPORTED_METHODS = 'saliency',
    ) -> np.ndarray:
        """
        Get selectivity scores for a given image.

        Parameters
        ----------
        image : torch.Tensor
            The image to be evaluated
        label : int
            The label of the image
        method : SUPPORTED_METHODS, optional
            The method to attribute by, by default 'saliency'

        Returns
        -------
        np.ndarray
            Selectivity scores
        """

        saliency_scores = self._get_saliency_map(image, label, method)

        # add dimension if saliency_scores is 2D (e.g. for score cam -> grey scale)
        if len(saliency_scores.shape) == 2:
            saliency_scores = saliency_scores[np.newaxis, ...]

        saliency_scores = self.saliency_resize(torch.Tensor(saliency_scores))

        if np.all(saliency_scores.numpy() == 0):
            return np.nan

        selectivity = self.selectivity(
            model=self.model,
            channel_first=True,
            x_batch=image.unsqueeze(0).numpy(),
            y_batch= torch.Tensor([label]).type(torch.int64).numpy(),
            device=self.device,
            a_batch=saliency_scores.unsqueeze(0).numpy(),
        )

        return np.trapz(selectivity, dx=1/len(selectivity[0]))
