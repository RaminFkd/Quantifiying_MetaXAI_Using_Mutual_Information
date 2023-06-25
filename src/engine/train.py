
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from tqdm import tqdm

def train_one_epoch(
    model: nn.Module,
    dataset: datasets.VisionDataset,
    batch_size: int,
    device: torch.device = torch.device('cpu'),
) -> defaultdict:
    """
    Train the model for one epoch. Using CrossEntropyLoss and Adam optimizer.
    Learning rate is 1e-3.
    Stop training if accuracy is at least 84.2% for 50 batches.

    Parameters
    ----------
    model : nn.Module
        The model to train
    dataset : datasets.VisionDataset
        The dataset to use
    batch_size : int
        The batch size to use

    Returns
    -------
    defaultdict
        The stats for the epoch
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
    )

    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    stats = defaultdict(list)
    loss = float('inf')
    acc = 0
    pbar = tqdm(
        dataloader,
        desc=f'Training | Loss: {loss:.4f} | Acc: {acc:.4f}'
    )

    for _, batch in pbar:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        stats['loss'].append(loss.item())
        _, predicted = torch.max(output, 1)
        acc = (predicted == labels).sum().item() / len(labels)
        stats['acc'].append(acc)

        loss = sum(stats['loss'][-50:]) / len(stats['loss'][-50:])
        acc = sum(stats['acc'][-50:]) / len(stats['acc'][-50:])

        pbar.set_description(
            f'Training | Loss: {loss:.4f} | Acc: {acc:.4f}'
        )

    return stats
