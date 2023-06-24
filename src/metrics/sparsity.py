from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .metric_base import SUPPORTED_METHODS, MetricBase


class Sparsity(MetricBase):

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        normalize: bool = True,
        resize: Tuple[int, int] | None = None,
        out_path: Path | None = None
    ) -> None:
        super().__init__(model, device, normalize, resize, out_path)

    def __call__(
        self,
        image: torch.Tensor,
        label: int,
        method: SUPPORTED_METHODS = 'saliency',
    ) -> float:
        """
        Compute the sparsity of the saliency map for a given image.

        Parameters
        ----------
        image : torch.Tensor
            The image to compute the sparsity of
        label : int
            The label of the image
        method : SUPPORTED_METHODS, optional
            The attribution method, by default 'saliency'

        Returns
        -------
        float
            The sparsity of the saliency map
        """
        saliency_scores = self._get_saliency_map(image, label, method)
        saliency_scores = self._to_one_dim(saliency_scores).flatten()

        min_max_norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        saliency_scores = min_max_norm(saliency_scores)

        return 1/np.mean(saliency_scores)
