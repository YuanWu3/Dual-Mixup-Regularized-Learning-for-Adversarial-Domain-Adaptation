# Dual-Mixup-Regularized-Learning
Implementation of "Dual Mixup Regularized Learning for Adversarial Domain Adaptation" in Pytorch (ECCV 2020)

## Datasets
This folder contains the dataset in the same format as needed by our code. You need to modify the path of the image in each ".txt" file in data folder

## Requirements:
- PyTorch >= 0.4.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.1
- Python3
- Numpy
- argparse
- PIL

## Training
All the parameters are set as the same as parameters mentioned in the article. You can use the following commands to the tasks:

```
USPS->MNIST
python train.py --epochs 50 --task U2M

MNIST->USPS
python train.py --epochs 50 --task M2U
```

## Citation
If you use this code for your research, consider citing:


    @article{wu2020dual,
      title={Dual Mixup Regularized Learning for Adversarial Domain Adaptation},
      author={Wu, Yuan and Inkpen, Diana and El-Roby, Ahmed},
      journal={arXiv preprint arXiv:2007.03141},
      year={2020}
    }
