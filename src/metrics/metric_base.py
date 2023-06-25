from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from captum.attr import IntegratedGradients, LayerGradCam, Saliency
from pytorch_grad_cam import ScoreCAM

SUPPORTED_METHODS = Literal[
    "saliency",
    "ig",
    "gradcam",
    "scorecam"
]


class MetricBase():
    """
    Base class for all metrics. Contains common methods and attributes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        normalize: bool = True,
        resize: Optional[Tuple[int, int]] = None,
        out_path: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.out_path = out_path
        self.device = device

        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        self.transform = T.Compose([
            T.Resize(resize if resize else (224,224), antialias=True),
            T.Normalize(
                mean=[0.485, 0.456, 0.406] if normalize else [0, 0, 0],
                std=[0.229, 0.224, 0.225] if normalize else [1, 1, 1],
            ),
        ])
        self.inverse_transform = T.Compose([
            T.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225] if normalize else [0, 0, 0],
                std=[1/0.229, 1/0.224, 1/0.225] if normalize else [1, 1, 1],
            ),
            T.ToPILImage(),
        ])

    def _get_saliency_map(
        self,
        image: torch.Tensor,
        label: int,
        method: str,
    ) -> np.ndarray:
        """
        Returns the saliency map for a given image and label.

        Parameters
        ----------
        image : torch.Tensor
            The image to compute the saliency map for. Must be a 3D tensor.
            With shape (C, H, W).
        label : int
            The label of the image.
        method : str
            The method to use for computing the saliency map. Must be one of
            "saliency", "ig", "gradcam", "scorecam"

        Returns
        -------
        np.ndarray
            The saliency map.

        Raises
        ------
        ValueError
            If there is no conv3 layer in the model.
        NotImplementedError
            If the method is not supported.
        """
        self.model.eval()
        self.model.to(self.device)

        label = int(label)
        image = self.transform(image).unsqueeze(0).to(self.device)

        conv3 = [module for name, module in self.model.named_modules() \
            if "conv3" in name]

        if len(conv3) == 0:
            raise ValueError(
                "No conv3 layer found. Currently only ResNet50 is supported."
            )

        conv3 = conv3[-1]

        with torch.no_grad():
            if method == "saliency":
                saliency = Saliency(self.model)
                attribution = saliency.attribute(image, target=label)
            elif method == "ig":
                integrated_gradients = IntegratedGradients(self.model)
                attribution = integrated_gradients.attribute(image, target=label)
            elif method == "scorecam":
                score_cam = ScoreCAM(
                    self.model,
                    target_layers=[conv3],
                    use_cuda=True if self.device == torch.device('cuda') else False
                )
                attribution = score_cam(image)
            elif method == "gradcam":
                grad_cam = LayerGradCam(self.model, conv3)
                attribution = grad_cam.attribute(image, target=label)
            else:
                raise NotImplementedError(
                    f"Method {method} is not suppurted, "\
                    f"choose one of {SUPPORTED_METHODS}."
                )

            if isinstance(attribution, torch.Tensor):
                return attribution.squeeze().cpu().detach().numpy()

            attribution = attribution.squeeze()
            return attribution

    def _to_one_dim(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts a 3D tensor to a 2D tensor by
        taking the mean over all channels.

        Parameters
        ----------
        image : torch.Tensor
            The image to convert to 2D.

        Returns
        -------
        torch.Tensor
            The converted image.
        """
        if len(image.shape) == 4:
            image = image.squeeze(0)

        if len(image.shape) == 3:
            image = np.mean(image, axis=0)

        return image
