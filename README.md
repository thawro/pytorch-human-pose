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
Inference using the _ClassificationHRNet_ model trained on ImageNet dataset (1000 classes). The parameters configurable via CLI:
* `--inference.input_size` - smaller edge of the image will be matched to this number (default: 256)
* `--inference.ckpt_path` - checkpoint path (default: *pretrained/hrnet_32.pt*)

### ImageNet data
Run inference on ImageNet val split with default input_size (256)
```bash
python src/classification/bin/inference.py --mode "val"
```

with changed input size
```bash
python src/classification/bin/inference.py --mode "val" --inference.input_size=512
```

### Custom data
```bash
python src/classification/bin/inference.py --mode "custom" --dirpath "data/examples/classification"
```

### Example outputs (top-5 probs):

![coyote](https://github.com/thawro/pytorch-human-pose/assets/50373360/b2c98474-b4b8-43cd-9c59-92e6f7231199)
![fox](https://github.com/thawro/pytorch-human-pose/assets/50373360/d7339a42-ee80-4f28-ab55-a79adc5f3715)
![shark](https://github.com/thawro/pytorch-human-pose/assets/50373360/dc292445-7da5-4224-bf2c-82053443ca0c)
![whale](https://github.com/thawro/pytorch-human-pose/assets/50373360/a15f6a15-73e7-499c-a42e-31d7a50a21d9)


## Human Pose (_HigherHRNet_)
Inference using the _HigherHRNet_ model trained on COCO keypoints dataset (17 keypoints). The parameters configurable via CLI:
* `--inference.input_size` - smaller edge of the image will be matched to this number (default: 256)
* `--inference.ckpt_path` - checkpoint path (default: *pretrained/higher_hrnet_32.pt*)
* `--inference.det_thr` - detection threshold used in grouping (default: 0.05)
* `--inference.tag_thr` - associative embedding tags threshold used in grouping (default: 0.5)
* `--inference.use_flip` - whether to use horizontal flip and average the results (default: False)


### COCO data
Run inference on COCO val split with default inference parameters
```bash
python src/keypoints/bin/inference.py --mode "val"
```

with changed input_size, use_flip and det_thr
```bash
python src/keypoints/bin/inference.py --mode "val" --inference.input_size=256 --inference.use_flip=True --inference.det_thr=0.1
```

### Custom data
```bash
python src/keypoints/bin/inference.py --mode "custom" --path "data/examples/keypoints/"
```

### Video
```bash
python src/keypoints/bin/inference.py --mode "custom" --path "data/examples/keypoints/simple_3.mp4"
```

### Example outputs (images)

Each sample is composed of Connections plot, Associative Embeddings visualization (after grouping) and Heatmaps plot 
1. Baseball
![AE_baseball](https://github.com/thawro/pytorch-human-pose/assets/50373360/6b18c18d-6d4b-4964-83c2-a546ee2baf5d)
![HM_baseball](https://github.com/thawro/pytorch-human-pose/assets/50373360/b446ae1f-0655-41f4-b968-8f29dc7e75b9)

2. Jump
![AE_jump](https://github.com/thawro/pytorch-human-pose/assets/50373360/434222d6-b3c4-464f-9e6b-bebd692fce41)
![HM_jump](https://github.com/thawro/pytorch-human-pose/assets/50373360/964fb30b-4c1f-43b6-b3ff-20c7be17cea8)

3. Park
![AE_park](https://github.com/thawro/pytorch-human-pose/assets/50373360/2009d88d-ccb4-4cb6-ae9c-b139d62cc76a)
![HM_park](https://github.com/thawro/pytorch-human-pose/assets/50373360/06c42437-1336-43db-9ea2-dac38c411827)

### Example outputs (videos)
Each sample with two input_sizes variants

1. Two people

input_size=256
https://github.com/thawro/pytorch-human-pose/assets/50373360/27c07d1a-af1a-46bc-93ab-5d9063f1ccba

input_size=512
https://github.com/thawro/pytorch-human-pose/assets/50373360/374c5dc2-53cf-4552-8454-1aa44d3810fc

2. Three people 

input_size=256
https://github.com/thawro/pytorch-human-pose/assets/50373360/b873c086-6010-49d2-b056-9c9f4bbef4e1

input_size=512
https://github.com/thawro/pytorch-human-pose/assets/50373360/447497d3-1a16-4eed-af17-9a52215552d6


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
