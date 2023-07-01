import torch
import quantus
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import time

from .metric_base import MetricBase, SUPPORTED_METHODS


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
        norm_image = self.transform(image)
        results = {method: self.selectivity(
                                    model=self.model, 
                                    channel_first=True,
                                    x_batch=norm_image.unsqueeze(0).numpy(),
                                    y_batch= torch.Tensor([label]).type(torch.int64).numpy(),
                                    a_batch=None,
                                    device=self.device,
                                    explain_func=quantus.explain,
                                    explain_func_kwargs={"method": method}) for method in ["Saliency", "IntegratedGradients"]}
        
        
        self.selectivity.plot(results=results)
        start = time.time()
        saliency_scores = self._get_saliency_map(image, label, method)
        result = {method: self.selectivity(
            model=self.model, 
            channel_first=True,
            x_batch=norm_image.unsqueeze(0).numpy(),
            y_batch= torch.Tensor([label]).type(torch.int64).numpy(),
            device=self.device,
            a_batch=torch.Tensor(saliency_scores).unsqueeze(0).numpy(),
        )}
        print(f"Time taken: {time.time() - start}")
        return result
