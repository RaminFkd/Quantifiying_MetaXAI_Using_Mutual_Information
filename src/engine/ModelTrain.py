
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

def train_one_epoch(
    model: nn.Module,
    dataset: datasets.VisionDataset,
    batch_size: int,
) -> defaultdict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    stop = False
    for idx, batch in pbar:
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

        if (len(stats['acc']) >= 50 and
                sum(stats['acc'][-50:])/len(stats['acc'][-50:]) >= 0.842):
            print("Reached 84.2%% accuracy. Ending training.")
            stop = True
            break

    return stats, stop
