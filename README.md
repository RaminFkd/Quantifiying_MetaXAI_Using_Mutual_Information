# Quantifying Redundant Meta-XAI Metrics Using Mutual Information

## Requirements

It is recommended to use a virtual environment to run the code. The code was tested using Python 3.10.*. To install the required packages, run the following command in the root directory of the repository:

```bash
pip install -r requirements.txt
```

## Usage

To generate Saliency Maps, run the following command:

```bash
cd src
python main.py [OPTIONS]
```

### Options
| Argument          | Type    | Default Value                    | Description                                                                                      |
|-------------------|---------|----------------------------------|--------------------------------------------------------------------------------------------------|
| --model           | str     | resnet50                         | Specifies the model to be used.                                                                  |
| --out_features    | int     | 200                              | Specifies the number of output features.                                                         |
| --pretrained      | boolean | False                            | Uses pretrained weights if set to True.                                                          |
| --weights         | str     | ./output/weights/resnet50_200.pth| Specifies the path to the weights file.                                                          |
| --dataset         | str     | CUB_200_2011                     | Specifies the dataset to be used for training.                                                   |
| --batch_size      | int     | 32                               | Specifies the batch size for training.                                                           |
| --cuda            | boolean | False                            | Uses GPU acceleration if set to True.                                                            |
| --run_analysis    | boolean | False                            | Performs analysis if set to True.                                                                |
| --run_mi          | boolean | False                            | Runs mutual information analysis if set to True.                                                 |
| --metrics         | str     | dauc,iauc,dc,ic,sparsity,selectivity| Specifies the metrics to be computed during analysis.                                           |
| --attribution     | str     | gradcam,scorecam,ig              | Specifies the attribution methods to be used during analysis.                                    |
| --out_dir         | str     | ./output/                         | Specifies the output directory for saving the results.                                           |
| --resize          | int     | 128                              | Specifies the size for resizing images.                                                          |
| --normalize       | boolean | False                            | Normalizes the input images if set to True.                                                      |
| --n               | int     | 200                              | Specifies for how many datapoints to run the experiment.                                                                        |

