[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/confidence-weighted-boundary-aware-learning/semi-supervised-semantic-segmentation-on-44)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-44?p=confidence-weighted-boundary-aware-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/confidence-weighted-boundary-aware-learning/semi-supervised-semantic-segmentation-on-3)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-3?p=confidence-weighted-boundary-aware-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/confidence-weighted-boundary-aware-learning/semi-supervised-semantic-segmentation-on-22)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-22?p=confidence-weighted-boundary-aware-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/confidence-weighted-boundary-aware-learning/semi-supervised-semantic-segmentation-on-27)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-27?p=confidence-weighted-boundary-aware-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/confidence-weighted-boundary-aware-learning/semi-supervised-semantic-segmentation-on-2)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-2?p=confidence-weighted-boundary-aware-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/confidence-weighted-boundary-aware-learning/semi-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-1?p=confidence-weighted-boundary-aware-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/confidence-weighted-boundary-aware-learning/semi-supervised-semantic-segmentation-on-4)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-4?p=confidence-weighted-boundary-aware-learning)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/confidence-weighted-boundary-aware-learning/semi-supervised-semantic-segmentation-on-9)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-9?p=confidence-weighted-boundary-aware-learning)

# CW-BASS: Confidence-Weighted Boundary Aware Learning for Semi-Supervised Semantic Segmentation

This is the official PyTorch implementation of our IJCNN 2025 paper:
> **CW-BASS: Confidence-Weighted Boundary Aware Learning for Semi-Supervised Semantic Segmentation**

![Fig. 2 Framework](docs/assets/images/Fig.%202%20Framework.png)

Project Page: https://psychofict.github.io/CW-BASS/

## Getting Started

### Data Preparation

#### Pre-trained Model

[ResNet-50](https://download.pytorch.org/models/resnet50-0676ba61.pth) | [DeepLabv2-ResNet-101](https://drive.google.com/file/d/14be0R1544P5hBmpmtr8q5KeRAvGunc6i/view?usp=sharing)

#### Dataset

[Pascal JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [Pascal SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing) | [Cityscapes leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [Cityscapes gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing) 

#### File Organization

```
├── ./pretrained
    ├── resnet50.pth
    └── deeplabv2_resnet101_coco_pretrained.pth
    
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
```


### Training and Testing

```
export semi_setting='pascal/1_8/split_0'

CUDA_VISIBLE_DEVICES=0,1 python -W ignore main.py \
  --dataset pascal --data-root [Your Pascal Path] \
  --batch-size 16 --backbone resnet50 --model deeplabv3plus \
  --labeled-id-path dataset/splits/$semi_setting/labeled.txt \
  --unlabeled-id-path dataset/splits/$semi_setting/unlabeled.txt \
  --pseudo-mask-path outdir/pseudo_masks/$semi_setting \
  --save-path outdir/models/$semi_setting
```


## Acknowledgement

The image partitions are borrowed from **Context-Aware-Consistency** and **PseudoSeg**. 
Part of the training hyper-parameters and network structures are adapted from **ST++** and **PyTorch-Encoding**. 

+ ST++: [https://github.com/LiheYoung/ST-PlusPlus](https://github.com/LiheYoung/ST-PlusPlus).
+ Context-Aware-Consistency: [https://github.com/dvlab-research/Context-Aware-Consistency](https://github.com/dvlab-research/Context-Aware-Consistency).
+ PyTorch-Encoding: [https://github.com/zhanghang1989/PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding).
+ OpenSelfSup: [https://github.com/open-mmlab/OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup).

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{tarubinga2025cw,
  title={CW-BASS: Confidence-Weighted Boundary-Aware Learning for Semi-Supervised Semantic Segmentation},
  author={Tarubinga, Ebenezer and Kalafatovich, Jenifer and Lee, Seong-Whan},
  journal={arXiv preprint arXiv:2502.15152},
  year={2025}
}
