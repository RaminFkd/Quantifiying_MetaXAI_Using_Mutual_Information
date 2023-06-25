import torch
import quantus
import numpy as np
from typing import Tuple,  Self, Optional
from pathlib import Path

from metric_base import MetricBase, SUPPORTED_METHODS


class Selectivity(MetricBase):

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        normalize: bool = True,
        resize: Optional[Tuple[int, int]] = None,
        out_path: Optional[Path] = None,
        patch_size: int = 56,
        perturb_baseline: str = "black"
    ) -> Self:

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
        return self.selectivity(
            model=self.model,
            a_batch=saliency_scores,
            device=self.device,
        )
