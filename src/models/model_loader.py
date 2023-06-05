import os
from pathlib import PurePath
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models

def load_cnn(
    model_name: str,
    pretrained: bool,
    out_features: int,
    load_weights: Optional[str],
) -> nn.Module:
    """
    Load a CNN model from torchvision.models and replace
    the last layer if possible

    Parameters
    ----------
    model_name : str
        The name of the model to load
    pretrained : bool
        Whether to load the pretrained model
    out_features : int
        The number of output features of the last layer
    load_weights : Optional[str]
        The path to the weights to load

    Returns
    -------
    nn.Module
        The loaded model

    Raises
    ------
    ValueError
        If the model name is not found in torchvision.models
    """

    # Check if the model name is a model in models
    if model_name not in models.__dict__:
        raise ValueError(f"Model {model_name} not found in torchvision.models")

    # Load the model
    pretrained = "DEFAULT" if pretrained else None
    model = models.__dict__[model_name](weights=pretrained)

    # Replace the last layer
    if model.__class__.__name__ == "ResNet":
        model.fc = nn.Linear(model.fc.in_features, out_features)
    elif model.__class__.__name__ == "DenseNet":
        model.classifier = nn.Linear(
            model.classifier.in_features,
            out_features
        )
    elif model.__class__.__name__ == "VGG":
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            out_features
        )
    elif model.__class__.__name__ == "SqueezeNet":
        model.classifier[1] = nn.Conv2d(
            model.classifier[1].in_channels,
            out_features,
            kernel_size=(1, 1),
            stride=(1, 1)
        )
    elif model.__class__.__name__ == "Inception3":
        model.fc = nn.Linear(model.fc.in_features, out_features)
    elif model.__class__.__name__ == "AlexNet":
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            out_features
        )
    elif model.__class__.__name__ == "ConvNeXt":
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            out_features
        )
    else:
        print(
            f"""Model {model_name} not supported,
            please modify out_features manually."""
        )

    # Load the weights
    tuned = False
    if load_weights is not None and os.path.exists(load_weights):
        load_weights = PurePath(load_weights)
        model.load_state_dict(torch.load(open(load_weights, 'rb')))
        tuned = True

    return model, tuned
