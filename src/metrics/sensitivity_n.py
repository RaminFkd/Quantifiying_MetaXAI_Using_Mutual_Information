import time
import torch
import quantus
import numpy as np

from typing import Tuple, Optional
from pathlib import Path

from .metric_base import MetricBase, SUPPORTED_METHODS


class SensitivityN(MetricBase):

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        normalize: bool = True,
        resize: Optional[Tuple[int, int]] = None,
        out_path: Optional[Path] = None,
        feature_in_step: int = 16,
        n_max_percentage: float = 0.3,
        similarity_func: quantus.similarity_func = quantus.similarity_func.correlation_pearson,
        perturb_func: quantus.perturb_func = quantus.perturb_func.baseline_replacement_by_indices,
        perturb_baseline: str = "uniform",
        return_aggregate: bool = False
    ) -> None:
        super().__init__(
            model,
            device,
            normalize,
            resize,
            out_path)

        self.sensitivity_n = quantus.SensitivityN(
            features_in_step=feature_in_step,
            n_max_percentage=n_max_percentage,
            similarity_func=similarity_func,
            perturb_func=perturb_func,
            perturb_baseline=perturb_baseline,
            return_aggregate=return_aggregate,
        )

    def __call__(
        self,
        image: torch.Tensor,
        label: int,
        method: SUPPORTED_METHODS = 'saliency',
    ) -> np.ndarray:
        """return correlation coefficients between input features and saliency scores

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
            Correlation coefficients between input features and saliency scores
        """
        norm_image = self.transform(image)
        results = self.sensitivity_n(model=self.model, 
                        x_batch=norm_image.unsqueeze(0).numpy(),
                        channel_first=True,
                        y_batch=torch.Tensor([label]).type(torch.int64).numpy(),
                        a_batch=None,
                        device=self.device,
                        explain_func=quantus.explain, 
                        explain_func_kwargs={"method": "IntegratedGradients"})
        
        start = time.time() 
        saliency_scores = self._get_saliency_map(image, label, method)
        result =  self.sensitivity_n(
            model=self.model,
            channel_first=True,
            x_batch=norm_image.unsqueeze(0).numpy(),
            y_batch= torch.Tensor([label]).type(torch.int64).numpy(),
            a_batch=torch.Tensor(saliency_scores).unsqueeze(0).numpy(),
            device=self.device,
        )
        print(f"Time taken: {time.time() - start}")
        return result
