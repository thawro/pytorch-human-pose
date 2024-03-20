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
make env
``` 

5. Create directories for training/inference purposes
```bash
make dirs
```


# Prerequisites

## Data preparation

> **_NOTE:_** The data preparation scripts use `tqdm` to show progress bars for file unzipping, so make sure to install the [Environment](#environment) first.

### ImageNet

1. Download dataset from [kagle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
   1. Go to [link](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
   2. Sign in to kaggle
   3. Scroll down and click "**Download All**" button
2. Move downloaded `imagenet-object-localization-challenge.zip` file to `<project_root>/data` directory
3. Run ImageNet preparation script from the `<project_root>` directory (it may take a while)
```bash
make imagenet
```

The script will unzip the downloaded `imagenet-object-localization-challenge.zip` file, remove it, create the `ImageNet` directory and move unzipped files from `ILSVRC/Data/CLS-LOC` directory to `ImageNet` directory. Then it will move the val image files to separate directories (named by wodnet labels) using [this](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) script and it will download the json mapping for ImageNet labels (from [this](https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json) source)

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
1. Run COCO preparation script from the `<project_root>` directory (it may take a while)
```bash
make coco
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

> **_NOTE:_** [Environment](#environment) must be installed to perform the inference

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

https://github.com/thawro/pytorch-human-pose/assets/50373360/c6c477e5-df4c-48c3-84e9-0c050cb26880

https://github.com/thawro/pytorch-human-pose/assets/50373360/6dc84f08-5c1a-4526-9fad-d055d20b1e53


2. Three people 

https://github.com/thawro/pytorch-human-pose/assets/50373360/9a9be24e-2143-4246-85d0-434fec4db301

https://github.com/thawro/pytorch-human-pose/assets/50373360/184c2071-417e-4380-bf81-99397704843e


# Training from scratch
> **_NOTE:_** [Environment](#environment) must be installed to perform the training

> **_IMPORTANT:_** MLFlow logging is enabled by default, so before every training one must run `make mlflow_server` to start the server

Most important training configurable CLI parameters (others can be checked in config python files):

* `setup.ckpt_path` - Path to checkpoint file saved during training (for training resume)
* `setup.pretrained_ckpt_path` - Path to checkpoint file with pretrained network weights
* `trainer.accelerator` - Device for training (`cpu` or `gpu`) 
* `trainer.limit_batches` - How many batches are used for training. Parameter used to run a debug experiment. When limit_batches > 0, then experiment is considered as debug
* `trainer.use_DDP` - Whether to run Distributed Data Parallel (DDP) training on multiple GPUs
* `trainer.sync_batchnorm` - Whether to use SyncBatchnorm class for DDP training

## Backbone
> **_NOTE:_** [ImageNet](#imagenet) data must be prepared to train the backbone model

### Pretraining _ClassificationHRNet_ on ImageNet

#### Using single GPU

```bash
python src/classification/bin/train.py --setup.ckpt_path=None --trainer.use_DDP=False
```
`--setup.ckpt_path=None` to ensure that new experiment is created, `--trainer.use_DDP=False` to ensure that single GPU is used

#### Using multiple GPUs - use [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html)
```bash
torchrun --standalone --nproc_per_node=2 src/classification/bin/train.py --setup.ckpt_path=None --trainer.use_DDP=True
```


### Evaluation of _ClassificationHRNet_ on Imagenet
TODO

## Human Pose
> **_NOTE:_** [COCO](#coco) data must be prepared to train the human pose model


### Training _HigherHRNet_ on COCO
#### Using single GPU

```bash
python src/keypoints/bin/train.py --setup.ckpt_path=None --trainer.use_DDP=False --setup.pretrained_ckpt_path="pretrained/hrnet_32.pt"
```
`--setup.ckpt_path=None` to ensure that new experiment is created, `--trainer.use_DDP=False` to ensure that single GPU is used, `--setup.pretrained_ckpt_path` to load pretrained backbone model from `hrnet_32.pt` file

#### Using multiple GPUs - use [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html)
```bash
torchrun --standalone --nproc_per_node=2 src/keypoints/bin/train.py --setup.ckpt_path=None --trainer.use_DDP=True --setup.pretrained_ckpt_path="pretrained/hrnet_32.pt"                
```


### Evaluation of _HigherHRNet_ on COCO (val2017)

> **_NOTE:_** Before running evaluation script you must ensure that correct `run_path` is defined inside the script. `run_path` must point to the directory where training checkpoint (`.pt` file) and config (`.yaml`) files are present.

```bash
python src/keypoints/bin/eval.py
```

After running this script there will be a `evaluation_results` directory created ( inside the `run_path` directory) with the evaluation output files:
* `coco_output.txt` - file with txt output from pycocotools (the table)
* `config.yaml` - config of the evaluated run
* `val2017_results.json` - json file with results (predicted keypoints coordinates)

---

Evaluation results obtained for inference parameters:
* `--inference.input_size=512`
* `--inference.use_flip=True`

| Metric name                                | Area    | Max Dets | Metric value |
| ------------------------------------------ | ------- | -------- | ------------ |
| **Average Precision  (AP) @IoU=0.50:0.95** | **all** | **20**   | **0.673**    |
| Average Precision  (AP) @IoU=0.50          | all     | 20       | 0.870        |
| Average Precision  (AP) @IoU=0.75          | all     | 20       | 0.733        |
| Average Precision  (AP) @IoU=0.50:0.95     | medium  | 20       | 0.615        |
| Average Precision  (AP) @IoU=0.50:0.95     | large   | 20       | 0.761        |
| Average Recall     (AR) @IoU=0.50:0.95     | all     | 20       | 0.722        |
| Average Recall     (AR) @IoU=0.50          | all     | 20       | 0.896        |
| Average Recall     (AR) @IoU=0.75          | all     | 20       | 0.770        |
| Average Recall     (AR) @IoU=0.50:0.95     | medium  | 20       | 0.652        |
| Average Recall     (AR) @IoU=0.50:0.95     | large   | 20       | 0.819        |


# Training code notes
TODO