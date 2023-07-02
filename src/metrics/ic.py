import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .iauc import IAUC
from .metric_base import SUPPORTED_METHODS, MetricBase


class IC(MetricBase):

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        normalize: bool = True,
        resize: Tuple[int, int] | None = None,
        out_path: Path | None = None
    ) -> None:
        super().__init__(model, device, normalize, resize, out_path)
        self.iauc = IAUC(model, device, normalize, resize, out_path)

    def __call__(
        self,
        image: torch.Tensor,
        label: int,
        method: SUPPORTED_METHODS = 'saliency',
        strict: bool = False,
        idx = None
    ) -> float:
        """
        Get IC score for a given image. The label is important to
        obtain attribution maps for the correct class.

        Parameters
        ----------
        image : torch.Tensor
            The image to be evaluated
        label : int
            The label of the image
        method : SUPPORTED_METHODS, optional
            The method to attribute by, by default 'saliency'
        strict : bool, optional
            If true, a meaningful value is only returned if the first predicted
            class matches the target, by default False
        idx : int, optional
            The index of the image, by default None

        Returns
        -------
        float
            The IC score
        """

        iauc_path = self.out_path / Path(f"iauc.pkl")
        if iauc_path.exists() and idx is not None:
            scores = self._load_scores(iauc_path, idx)
        else:
            scores = self.iauc(image, label, method, strict)['scores']

        if len(scores) == 0:
            return 0.0

        # Calculate variation (progressive difference) of the scores
        # and repeat last value to make variation and scores have
        # the same length
        variation = np.diff(scores)

        saliency_scores = self._get_saliency_map(image, label, method)
        saliency_scores = self._to_one_dim(saliency_scores).flatten()
        saliency_scores = np.sort(saliency_scores)

        # Calculate correlation between variation and saliency scores
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr = np.corrcoef(variation, saliency_scores)[0, 1]
        corr = corr if not np.isnan(corr) else 0.0

        return corr
