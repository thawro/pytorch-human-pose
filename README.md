# Multi Person Pose Estimation with PyTorch
[HigherHRNet](https://arxiv.org/abs/1908.10357) architecture implemented and trained from scratch using ImageNet and COCO datasets.
The model is trained in two steps:
1. Classification Backbone pretraining on ImageNet dataset (_ClassificationHRNet_)
2. Human Pose model Training on COCO (_HigherHRNet_)

# Environment
The environment management is handled with the use of `poetry`. To install the virtual environment:

1. Clone the repository
```bash
git clone https://github.com/thawro/pytorch-human-pose.git
```

2. Move to the repository (`<project_root>`)
```bash
cd pytorch-human-pose
```

3. Install `poetry` - follow [documentation](https://python-poetry.org/docs/#installation)

4. Install the virtual environment and activate venv
```bash
poetry install
poetry shell
``` 
or use one of the scripts:
```bash
bash scripts/prepare_environment.sh
```


# Prerequisites

## Data preparation

### ImageNet

1. Download dataset from [kagle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
2. Extract the downloaded zip file to `<project_root>/data` directory
3. Remove zip file
4. Run ImageNet preparation script (from the `<project_root>` directory)
```bash
bash scripts/prepare_imagenet.sh
```

The script will create the `ImageNet` directory and move the files from `ILSVRC/Data/CLS-LOC` directory to `ImageNet` directory. Then it will move the val image files to separate directories (named by wodnet labels) using [this](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) script and it will download json mapping for ImageNet labels (from [this](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json) source)

After these steps there should be a directory `data/ImageNet` with the following directory structure:

    data/ImageNet
    ├── wordnet_labels.yaml
    ├── train
    |   ├── n01440764
    |   ├── n01443537
    |   ...
    |   └── n15075141
    └── val
        ├── n01440764
        ├── n01443537
        ...
        └── n15075141

### COCO
1. Run COCO preparation script (from the <project_root> directory)
```bash
bash scripts/prepare_coco.sh
```

The script will create `data/COCO` directory, download files from the COCO [website](https://cocodataset.org/#download) (_2017 Train images [118K/18GB]_, _2017 Val images [5K/1GB]_, _2017 Test images [41K/6GB]_, _2017 Train/Val annotations [241MB]_) to the `data/COCO` directory, unzip the files, move the files to `images` and `annotations` subdirectories and remove the redundant zip files.

After these steps there should be a directory `data/COCO` with the following directory structure:

    data/COCO
    ├── annotations
    │   ├── captions_train2017.json
    │   ├── captions_val2017.json
    │   ├── instances_train2017.json
    │   ├── instances_val2017.json
    │   ├── person_keypoints_train2017.json
    │   └── person_keypoints_val2017.json
    └── images
        ├── test2017
        ├── train2017
        └── val2017

## Virtual environment installation
Install the poetry virtual environment following [Environment](#environment) steps.

## Checkpoints with trained models

The checkpoints are available at [Google Drive](https://drive.google.com/drive/folders/1srwrno9YYSBy1Wqf-uDok_TpID9Mm20d?usp=sharing):

* **`hrnet_32.pt`** - backbone pretrained on the ImageNet
* **`higher_hrnet_32.pt`** - pose estimation model trained on COCO

After download, place the checkpoints inside the **_pretrained_** directory.



# Inference

> **_NOTE:_** Checkpoints must be present in _pretrained_ directory to perform the inference.

## Classification (_ClassificationHRNet_)

### ImageNet data

## Human Pose (_HigherHRNet_)

### COCO data

### Video


# Training from scratch

## Backbone

### Pretraining _ClassificationHRNet_ on ImageNet

### Evaluation of _ClassificationHRNet_ on Imagenet


## Human Pose

### Training _HigherHRNet_ on COCO

### Evaluation of _HigherHRNet_ on COCO

**val2017**
| First Header | Second Header |
| ------------ | ------------- |
| Content Cell | Content Cell  |
| Content Cell | Content Cell  |
