from collections import defaultdict
import itertools
import pickle
import time
from argparse import ArgumentParser
from pathlib import Path
from threading import Thread
import numpy as np

import torch

from engine import train
from metrics import (DAUC, DC, IAUC, IC, Selectivity, SensitivityN, Sparsity,
                     continuous_mutual_info_score, continuous_mutual_info_gap)
from models import model_loader
from utils import dataset_loader

parser = ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--out_features', type=int, default=200)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--weights', type=str, default="./output/weights/resnet50_200.pth")

# Training arguments
parser.add_argument('--dataset', type=str, default='CUB_200_2011')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--cuda', action='store_true')

# Analysis arguments
parser.add_argument('--run_analysis', action='store_true')
parser.add_argument('--run_mi', action='store_true')
parser.add_argument('--metrics', type=str, default='dauc,iauc,dc,ic,sparsity,selectivity')
parser.add_argument('--attribution', type=str, default='gradcam,scorecam,ig')
parser.add_argument('--out_dir', type=str, default='./output/')
parser.add_argument('--resize', type=int, default=128)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--n', type=int, default=200)

args = parser.parse_args()


METRICS = {
    'dauc': DAUC,
    'iauc': IAUC,
    'dc': DC,
    'ic': IC,
    'sparsity': Sparsity,
    'selectivity': Selectivity,
    'sensitivity_n': SensitivityN,
}


class SaveWorker:

    def __init__(self, out_file: Path):
        self._out_file = out_file
        self._out_file.parent.mkdir(parents=True, exist_ok=True)
        self._buffer = []
        self._shutdown = False

        self._save_thread = Thread(target=self._save_daemon, daemon=True)
        self._save_thread.start()

    def shutdown(self):
        self._shutdown = True

    def save(self, obj):
        self._buffer.append(obj)

    def _flush(self):
        if self._out_file.exists():
            with open(self._out_file, 'rb') as f:
                data = pickle.load(f) + self._buffer
        else:
            data = self._buffer

        with open(self._out_file, 'wb') as f:
            pickle.dump(data, f)

    def _save_daemon(self):
        while not self._shutdown or len(self._buffer) > 0:
            if len(self._buffer) > 0:
                self._flush()
                self._buffer = []

            time.sleep(5)


def main_train(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: torch.device,
    weights_path: Path,
) -> list:
    stats = []
    for epoch in range(5):
        stats.append(
            train.train_one_epoch(
                model=model,
                dataset=dataset,
                batch_size=batch_size,
                device=device,
            )
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), open(weights_path, 'wb'))

    return stats


def main_metric_analysis(
    model: torch.nn.Module,
    attributions: list,
    metrics: list,
    device: torch.device,
    out_dir: Path,
    resize: tuple,
    normalize: bool,
):
    for attr in attributions:
        for metr in metrics:
            print(f"Running {attr} {metr}...")
            to_save = SaveWorker(out_dir / Path(f"{attr}/{metr}.pkl"))

            metric = METRICS[metr](
                model=model,
                device=device,
                normalize=normalize,
                resize=(resize, resize),
                out_path=out_dir / Path(f"{attr}/"),
            )

            itr = 0
            for idx, (img, label) in dataset:
                if itr >= args.n:
                    break

                val = metric(img, label, attr, idx=idx)
                to_save.save((idx, val))
                itr += 1

            to_save.shutdown()

def main_mi(
    out_dir: Path,
):
    files = defaultdict(list)
    for f in out_dir.glob('**/*.pkl'):
        files[f.parent.name].append(f)

    to_save = SaveWorker(out_dir / Path(f"mi.pkl"))
    for attr in files:
        for combination in itertools.combinations(files[attr], 2):
            print(f"Running MI on {combination[0]} and {combination[1]}...")

            with open(combination[0], 'rb') as f:
                x = pickle.load(f)
            with open(combination[1], 'rb') as f:
                y = pickle.load(f)

            x = [i[1] for i in x]
            y = [i[1] for i in y]

            if combination[0].stem == 'iauc' or combination[0].stem == 'dauc':
                x = [i[combination[0].stem] for i in x]
            if combination[1].stem == 'iauc' or combination[1].stem == 'dauc':
                y = [i[combination[1].stem] for i in y]

            # stack lists to numpy array or convert to numpy array
            if isinstance(x[0], np.ndarray):
                x = np.stack(x).squeeze()
            else:
                x = np.array(x)

            if isinstance(y[0], np.ndarray):
                y = np.stack(y).squeeze()
            else:
                y = np.array(y)

            mi_scores = continuous_mutual_info_gap(x, y, k=3)
            results = {
                'metrics': (combination[0].stem, combination[1].stem),
                'attribution': attr,
                'MI': mi_scores['MI'],
                'MIG_x': mi_scores['MIG_x'],
                'MIG_y': mi_scores['MIG_y'],
            }

            to_save.save(results)

    to_save.shutdown()

if __name__=='__main__':

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    model, tuned = model_loader.load_cnn(
        model_name=args.model,
        pretrained=args.pretrained,
        out_features=args.out_features,
        load_weights=Path(args.weights) if args.weights else None,
    )
    dataset = dataset_loader.load_dataset(args.dataset, (args.resize, args.resize))
    out_dir = Path(args.out_dir)
    weights_path = out_dir / 'weights' / f"{args.model}_{args.out_features}.pth"

    if not tuned:
        main_train(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            weights_path=weights_path,
        )

    metr_list = args.metrics.split(',')
    attr_list = args.attribution.split(',')

    if args.run_analysis:
        main_metric_analysis(
            model=model,
            attributions=attr_list,
            metrics=metr_list,
            device=device,
            out_dir=out_dir,
            resize=args.resize,
            normalize=args.normalize,
        )

    if args.run_mi:
        main_mi(
            out_dir=out_dir,
        )

