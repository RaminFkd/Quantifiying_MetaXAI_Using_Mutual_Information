import pickle
from argparse import ArgumentParser
from pathlib import Path

import torch

from engine import train
from metrics import (DAUC, DC, IAUC, IC, Selectivity, SensitivityN, Sparsity,
                     continuous_mutual_info_score)
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
        for epoch in range(5):
            stats = train.train_one_epoch(
                model=model,
                dataset=dataset,
                batch_size=args.batch_size,
                device=device,
            )
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), open(weights_path, 'wb'))

    metr_list = args.metrics.split(',')
    attr_list = args.attribution.split(',')

    to_save = []
    for metr in metr_list:
        for attr in attr_list:
            metric = METRICS[metr](
                model=model,
                device=device,
                normalize=args.normalize,
                resize=(args.resize, args.resize),
            )

            itr = 0
            for idx, (img, label) in dataset:
                if itr >= args.n:
                    break
                to_save.append((idx, metric(img, label, attr)))
                itr += 1

            metr_out = out_dir / Path(f"{attr}/{metr}.pkl")
            metr_out.parent.mkdir(parents=True, exist_ok=True)
            with open(metr_out, 'wb') as f:
                pickle.dump(to_save, f)
