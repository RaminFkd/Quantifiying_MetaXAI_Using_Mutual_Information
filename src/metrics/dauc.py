from pathlib import Path
from typing import Any, Dict, List, Tuple, get_args

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metric_base import SUPPORTED_METHODS, MetricBase


class DAUC(MetricBase):

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
        strict: bool = False,
        idx: int = None
    ) -> Dict[str, Any]:
        """
        Compute the Deletion AUC (DAUC) for a given image and label. The
        label is used for the attribution method to compute the saliency map
        for the correct class.

        Parameters
        ----------
        image : torch.Tensor
            Image to compute the DAUC for.
        label : int
            The label of the image.
        method : SUPPORTED_METHODS, optional
            The method to attribute by, by default 'saliency'
        strict : bool, optional
            If true, a meaningful value is only returned if the first predicted
            class matches the target, by default False
        idx : int, optional
            The index of the image, by default None

        Returns
        -------
        Dict[str, Any]
            A dictionary with the keys 'dauc', 'scores', 'prediction'
            and 'target'. The value of 'dauc' is the DAUC score, 'scores' is
            the list of scores for each masking step, 'prediction' is a tuple of
            the predicted class and its probability, and 'target' is a tuple of
            the target class and its probability.
        """

        assert method in get_args(SUPPORTED_METHODS), \
            f'Invalid method: {method} not in {get_args(SUPPORTED_METHODS)}'

        dauc_scores = []

        initial_prediction = self._initial_prediction(image)
        masked_predictions = self._masked_predictions(image, label, method)

        pred_label = np.argmax(initial_prediction)

        if strict and pred_label != label:
            return {
                'dauc': 0.0,
                'scores': [],
                'prediction': (
                    pred_label,
                    F.softmax(torch.tensor(initial_prediction), dim=0)[pred_label]
                ),
                'target': (
                    label,
                    F.softmax(torch.tensor(initial_prediction), dim=0)[label]
                ),
            }

        probs = F.softmax(torch.tensor(initial_prediction), dim=0)
        dauc_scores.append(probs[pred_label])
        for masked_prediction in masked_predictions:
            probs = F.softmax(torch.tensor(masked_prediction), dim=0)
            dauc_scores.append(probs[pred_label])

        x = np.linspace(0, 1, len(dauc_scores))
        dauc = np.trapz(dauc_scores, x)

        return {
            'dauc': dauc,
            'scores': dauc_scores,
            'prediction': (
                pred_label,
                F.softmax(torch.tensor(initial_prediction), dim=0)[pred_label]
            ),
            'target': (
                label,
                F.softmax(torch.tensor(initial_prediction), dim=0)[label]
            ),
        }

    def _initial_prediction(
        self,
        image: torch.Tensor,
    ) -> np.ndarray:
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            out = self.model(image)

        return out.squeeze().cpu().detach().numpy()

    def _masked_predictions(
        self,
        image: torch.Tensor,
        label: int,
        method: str,
    ) -> List[np.ndarray]:
        image = self.transform(image)
        saliency_map = self._get_saliency_map(image, label, method)
        saliency_map = self._to_one_dim(saliency_map)
        # Get the ratio of the image to the saliency map
        tau = image.shape[1]/saliency_map.shape[0]
        assert tau == image.shape[2]/saliency_map.shape[1], \
            "Image and saliency map must have the same aspect ratio"

        # Get indices from highest to lowest saliency
        indices = np.unravel_index(
            np.argsort(saliency_map.ravel(), axis=None),
            saliency_map.shape
        )[::-1]
        mask = np.ones(image.shape[1:], dtype=np.int8)
        predictions = []

        # Iterate over the indices and mask the image at each index
        with torch.no_grad():
            self.model.eval()
            for idx in tqdm(zip(*indices), desc='DAUC - Masking image'):
                i, j = idx
                mask[int(i*tau):int((i+1)*tau), int(j*tau):int((j+1)*tau)] = 0
                masked_image = image * mask
                out = self.model(masked_image.unsqueeze(0).to(self.device))
                predictions.append(out.squeeze().cpu().detach().numpy())

        return predictions
