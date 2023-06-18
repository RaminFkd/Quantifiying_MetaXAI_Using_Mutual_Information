import pickle as pkl
from pathlib import Path
import numpy as np

import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from tqdm import tqdm

from models import model_loader
from utils import metrics


WEIGHTS_PATH = './output/weights/resnet50_9.pth'
PATH_TO_MAPS = Path('./output/')


def run_iauc_dauc_inference(
    model: torch.nn.Module,
    path_to_maps: Path,
    dev: torch.device,
):
    maps = metrics.load_maps(path_to_maps=path_to_maps)
    out_path = path_to_maps.parent / Path('dauc_iauc.pkl')
    if out_path.exists():
        open(out_path, 'wb').close()

    dauc = metrics.DAUC(
        model=model,
        maps=maps,
        device=dev,
        verbose=False
    )
    iauc = metrics.IAUC(
        model=model,
        maps=maps,
        device=dev,
        verbose=False
    )

    img_idx = 0
    for d_auc, i_auc in tqdm(
        zip(dauc, iauc),
        total=99,
        desc='Calculating DAUC and IAUC'
    ):
        d_auc_val, d_scores = d_auc
        i_auc_val, i_scores = i_auc
        x = [i for i in range(len(d_scores))]
        x = [i / (len(d_scores)-1) for i in x.copy()]

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title(f'DAUC {d_auc_val:.6f}')
        ax[0].plot(x, d_scores, color='r', linewidth=0.5)
        ax[0].set_xlabel('Proportion of masked pixels')
        ax[0].set_ylabel('Normalized score')
        ax[0].fill_between(x, d_scores, alpha=0.3, color='r')
        ax[0].set_ylim([0, 1])
        ax[0].set_xlim([0, 1])

        ax[1].set_title(f'IAUC {i_auc_val:.6f}')
        ax[1].plot(x, i_scores, color='b', linewidth=0.5)
        ax[1].set_xlabel('Proportion of masked pixels')
        ax[1].set_ylabel('Normalized score')
        ax[1].fill_between(x, i_scores, alpha=0.3, color='b')
        ax[1].set_ylim([0, 1])
        ax[1].set_xlim([0, 1])

        fig.tight_layout()
        parent = path_to_maps.parent
        fig.savefig(parent / f'dauc_iauc_{img_idx}.png')
        plt.close()

        with open(out_path, 'ab') as f:
            pkl.dump((img_idx, d_auc, i_auc), f)

        img_idx += 1

        if img_idx == 100:
            break


def run_dc_ic_sparsity_inference(
    model: torch.nn.Module,
    path_to_maps: Path,
    dev: torch.device,
):
    maps = metrics.load_maps(path_to_maps=path_to_maps)

    path_to_scores = path_to_maps.parent / Path('dauc_iauc.pkl')
    dauc_iauc = metrics.load_dauc_iauc(path_to_scores)

    scores = {
        'dauc': np.array([score_tuple[1] for score_tuple in dauc_iauc['dauc']]),
        'iauc': np.array([score_tuple[1] for score_tuple in dauc_iauc['iauc']]),
        'idx': dauc_iauc['idx']
    }

    dc = metrics.DC(
        model=model,
        maps=maps,
        device=dev,
        dauc_scores=scores,
        verbose=False
    ).get_dc()

    ic = metrics.IC(
        model=model,
        maps=maps,
        device=dev,
        iauc_scores=scores,
        verbose=False
    ).get_ic()

    saliency_maps = [maps[i][1] for i in range(len(dc))]
    saliency_maps = np.array(saliency_maps)
    sparsity = metrics.Sparsity().get_all_sparsities(
        saliency_maps = saliency_maps,
    )

    out_path = path_to_maps.parent / Path('dc_ic.pkl')
    with open(out_path, 'wb') as f:
        pkl.dump({'DC': dc, 'IC': ic, 'Sparsity': sparsity}, f)

    return dc, ic

if __name__=='__main__':

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {DEVICE}.')

    print('Loading model...')
    MODEL, _ = model_loader.load_cnn(
        model_name='resnet50',
        pretrained=True,
        out_features=200,
        load_weights=WEIGHTS_PATH,
    )
    MODEL.to(DEVICE)
    MODEL.eval()

    for map_path in PATH_TO_MAPS.glob('**/*_maps.pkl'):
        print(f'Processing {map_path.name}...')

        #check if dauc_iauc.pkl exists, if not run inference
        if not (map_path.parent / Path('dauc_iauc.pkl')).exists():
            run_iauc_dauc_inference(
                model=MODEL,
                path_to_maps=map_path,
                dev=DEVICE
            )

        run_dc_ic_sparsity_inference(
            model=MODEL,
            path_to_maps=map_path,
            dev=DEVICE
        )

    print('Done.')

