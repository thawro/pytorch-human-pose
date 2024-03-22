

# Multi Person Pose Estimation with PyTorch
[HigherHRNet](https://arxiv.org/abs/1908.10357) architecture implemented and trained from scratch using ImageNet and COCO datasets.
The model is trained in two steps:
1. Classification Backbone pretraining on ImageNet dataset (_ClassificationHRNet_ model)
2. Human Pose model Training on COCO dataset (_HigherHRNet_ model)

---

<details> 

<summary><b><font size="+2">📜 Table of Contents</font></b></summary>

- [💻 Environment](#environment)
- [🎒 Prerequisites](#-prerequisites)
  - [Data preparation](#data-preparation)
    - [ImageNet](#imagenet)
    - [COCO](#coco)
  - [Virtual environment installation](#virtual-environment-installation)
  - [Checkpoints with trained models](#checkpoints-with-trained-models)
- [📊 Inference](#-inference)
  - [Classification (_ClassificationHRNet_)](#classification-classificationhrnet)
    - [ImageNet data](#imagenet-data)
    - [Custom data](#custom-data)
    - [Example outputs (top-5 probs):](#example-outputs-top-5-probs)
  - [Human Pose (_HigherHRNet_)](#human-pose-higherhrnet)
    - [COCO data](#coco-data)
    - [Custom data](#custom-data-1)
    - [Video](#video)
    - [Example outputs (images)](#example-outputs-images)
    - [Example outputs (videos)](#example-outputs-videos)
- [📉 Training from scratch](#-training-from-scratch)
  - [Backbone](#backbone)
    - [Pretraining _ClassificationHRNet_ on ImageNet](#pretraining-classificationhrnet-on-imagenet)
      - [Using single GPU](#using-single-gpu)
      - [Using multiple GPUs - use `torchrun`](#using-multiple-gpus---use-torchrun)
    - [Evaluation of _ClassificationHRNet_ on Imagenet](#evaluation-of-classificationhrnet-on-imagenet)
  - [Human Pose](#human-pose)
    - [Training _HigherHRNet_ on COCO](#training-higherhrnet-on-coco)
      - [Using single GPU](#using-single-gpu-1)
      - [Using multiple GPUs - use `torchrun`](#using-multiple-gpus---use-torchrun-1)
    - [Evaluation of _HigherHRNet_ on COCO (val2017)](#evaluation-of-higherhrnet-on-coco-val2017)
- [📚 Training code guide](#-training-code-guide)
  - [Code structure](#code-structure)
  - [Code Guide](#code-guide)
    - [Configs](#configs)
    - [Training](#training)
    - [MLFlow](#mlflow)

</details>

---

# 💻 Environment
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

4. Install the virtual environment and activate it (the script runs `poetry install` and `poetry shell`)
```bash
make env
``` 

5. Create directories for training/inference purposes
```bash
make dirs
```


# 🎒 Prerequisites 
> **_NOTE:_** If you have installed the environment already (with `make env` or `poetry install`) you can activate it with `poetry shell`.

## Data preparation

> **_NOTE:_** The data preparation scripts use `tqdm` to show progress bars for file unzipping, so make sure to install and activate the [Environment](#environment) first.

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



# 📊 Inference

> **_NOTE:_** Checkpoints must be present in _pretrained_ directory to perform the inference.

> **_NOTE:_** You must first install and activate the [Environment](#environment) to perform the inference.

## Classification (_ClassificationHRNet_)
Inference using the _ClassificationHRNet_ model trained on ImageNet dataset (1000 classes). The parameters configurable via CLI:
* `--inference.input_size` - smaller edge of the image will be matched to this number (default: 256)
* `--inference.ckpt_path` - checkpoint path (default: *pretrained/hrnet_32.pt*)

### ImageNet data

> **_NOTE:_** [ImageNet](#imagenet) data must be prepared to perform inference on it.

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

<details>
<summary>👉 more examples</summary>

![fox](https://github.com/thawro/pytorch-human-pose/assets/50373360/d7339a42-ee80-4f28-ab55-a79adc5f3715)
![shark](https://github.com/thawro/pytorch-human-pose/assets/50373360/dc292445-7da5-4224-bf2c-82053443ca0c)
![whale](https://github.com/thawro/pytorch-human-pose/assets/50373360/a15f6a15-73e7-499c-a42e-31d7a50a21d9)

</details>

## Human Pose (_HigherHRNet_)
Inference using the _HigherHRNet_ model trained on COCO keypoints dataset (17 keypoints). The parameters configurable via CLI:
* `--inference.input_size` - smaller edge of the image will be matched to this number (default: 256)
* `--inference.ckpt_path` - checkpoint path (default: *pretrained/higher_hrnet_32.pt*)
* `--inference.det_thr` - detection threshold used in grouping (default: 0.05)
* `--inference.tag_thr` - associative embedding tags threshold used in grouping (default: 0.5)
* `--inference.use_flip` - whether to use horizontal flip and average the results (default: False)


### COCO data

> **_NOTE:_** [COCO](#coco) data must be prepared to perform inference on it.

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

<details>
<summary>👉 More examples</summary>

2. Jump
![AE_jump](https://github.com/thawro/pytorch-human-pose/assets/50373360/434222d6-b3c4-464f-9e6b-bebd692fce41)
![HM_jump](https://github.com/thawro/pytorch-human-pose/assets/50373360/964fb30b-4c1f-43b6-b3ff-20c7be17cea8)

3. Park
![AE_park](https://github.com/thawro/pytorch-human-pose/assets/50373360/2009d88d-ccb4-4cb6-ae9c-b139d62cc76a)
![HM_park](https://github.com/thawro/pytorch-human-pose/assets/50373360/06c42437-1336-43db-9ea2-dac38c411827)

</details>


### Example outputs (videos)
Each sample with two input_sizes variants

1. Two people (size: 256)

https://github.com/thawro/pytorch-human-pose/assets/50373360/c6c477e5-df4c-48c3-84e9-0c050cb26880

<details>
<summary>👉 More examples</summary>

2. Two people (size: 512)

https://github.com/thawro/pytorch-human-pose/assets/50373360/6dc84f08-5c1a-4526-9fad-d055d20b1e53

3. Three people  (size: 256)

https://github.com/thawro/pytorch-human-pose/assets/50373360/9a9be24e-2143-4246-85d0-434fec4db301


4. Three people  (size: 512)

https://github.com/thawro/pytorch-human-pose/assets/50373360/184c2071-417e-4380-bf81-99397704843e

</details>

</br>

# 📉 Training from scratch

> **_NOTE:_** You must first install and activate the [Environment](#environment) to perform the training.

> **_IMPORTANT:_** MLFlow logging is enabled by default, so before every training one must run `make mlflow_server` to start the server.

Most important training configurable CLI parameters (others can be checked in config python files):

* `setup.ckpt_path` - Path to checkpoint file saved during training (for training resume)
* `setup.pretrained_ckpt_path` - Path to checkpoint file with pretrained network weights
* `trainer.accelerator` - Device for training (`cpu` or `gpu`) 
* `trainer.limit_batches` - How many batches are used for training. Parameter used to run a debug experiment. When limit_batches > 0, then experiment is considered as debug
* `trainer.use_DDP` - Whether to run Distributed Data Parallel (DDP) training on multiple GPUs
* `trainer.sync_batchnorm` - Whether to use SyncBatchnorm class for DDP training

## Backbone
> **_NOTE:_** [ImageNet](#imagenet) data must be prepared to train the backbone model.

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

> **_NOTE:_** [COCO](#coco) data must be prepared to train the human pose model.


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


# 📚 Training code guide 

## Code structure

    .
    ├── data                    # datasets files
    │   ├── COCO                #   COCO dataset
    │   ├── examples            #   example inputs for inference
    │   └── ImageNet            #   ImageNet dataset
    |
    ├── experiments             # experiments configs - files needed to perform training/inference
    │   ├── classification      #   configs for ClassificationHRNet
    │   └── keypoints           #   configs for HigherHRNet
    |
    ├── inference_out           # directory with output from inference
    │   ├── classification      #   classification inference output
    │   └── keypoints           #   keypoints inference output
    |
    ├── Makefile                # Makefile for cleaner scripts using
    |
    ├── mlflow                  # mlflow files
    │   ├── artifacts           #   artifacts saved during training
    │   ├── mlruns.db           #   database for mlflow metrics saved during training
    │   └── test_experiment.py  #   script for some mlflow server testing
    |
    ├── poetry.lock             # file updated during poetry environment management
    |
    ├── pretrained              # directory with trained checkpoints
    │   ├── higher_hrnet_32.pt  #   HigherHRNet checkpoint - COCO human pose model
    │   └── hrnet_32.pt         #   ClassificationHRNet checkpoint - ImageNet classification model
    |
    ├── pyproject.toml          # definition of poetry environment
    |
    ├── README.md               # project README
    |
    ├── RESEARCH.md             # my sidenotes for human pose estimation task
    |
    ├── results                 # directory with training results/logs
    │   ├── classification      #   classification experiment results
    │   ├── debug               #   debug experiments results
    │   └── keypoints           #   keypoints experiment results
    |
    ├── scripts                 # directory with useful scripts
    │   ├── prepare_coco.sh     #   prepares COCO dataset - can be used without any other actions 
    │   ├── prepare_dirs.sh     #   creates needed directories
    │   ├── prepare_env.sh      #   installs and activates poetry environment
    │   ├── prepare_imagenet.sh #   prepares ImageNet dataset - requires ImageNet zip file to be downloaded before running
    │   └── run_mlflow.sh       #   runs mlflow server (locally)
    |
    └── src                     # project modules
        ├── base                #   base module - defines interfaces, abstract classes and useful training loops
        ├── classification      #   classification related files subclasses
        ├── keypoints           #   keypoints related files subclasses
        ├── logger              #   logging functionalities (monitoring and training loggers)
        └── utils               #   utilities functions (files loading, images manipulation, configs parsing, etc.)

## Code Guide

### Configs

Training and inference is parametrized using configs. Configs are defined in `experiments` directory using the `.yaml` files.
`.yaml` files parsing is done with dataclasses tailored for this purpose.`classification` and `keypoints` configs share some custom 
implementations which are defined in `src/base/config.py`. Task specific configs are implemented in `src/classification/config.py` and `src/keypoints/config.py`.
The Config dataclasses allow to overwrite the config parameters loaded from `.yaml` files by putting additional arguments to script calls using the following notation:
`--<field_name>.<nested_field_name>=<new_value>`, for example:

```bash
python src/keypoints/bin/train.py --setup.ckpt_path=None --trainer.use_DDP=False --setup.pretrained_ckpt_path=None
```

overwrites the `setup.ckpt_path`, `trainer.use_DDP` and `setup.pretrained_ckpt_path` attributes. 

The Config dataclasses are also repsonsible for creation of training and inference related objects with the use of the following methods:
* `create_net` (_task-specific_) - create neural network object (`torch.nn.Module`)
* `create_datamodule` (_task-specific_) - create datamodule (object used for loading train/val/test data into batches)
* `create_module` (_task-specific_) - create training module (object used to handle training and validation steps)
* `create_inference_model` (_task-specific_) - create model tailored for inference purposes
* `create_callbacks` - create callbacks (objects used during the training, each with special hooks)
* `create_logger` - create logger (object used for logging purposes)
* `create_trainer` - create trainer (object used to manage the whole training pipeline)

### Training

> **_IMPORTANT:_** You must ensure that environment is active (`poetry shell`) and mlflow server is running (`make mlflow_server`) before training.

During training the `results` directory is being populated with useful info about runs (logs, metrics, evaluation examples, etc.).
The structure of the populated `results` directory is the following:

    results
    └── <experiment_name>               # run experiment_name (e.g. classification)
        └── <run_name>                  # run run_name (e.g. 03-21_11:05__ImageNet_ClassificationHRNet)
            ├── <timestamp_1>           # run timestamp (e.g. 03-21_11:05)
            |   ├── checkpoints         # saved checkpoints
            |   ├── config.yaml         # config used for current run
            |   ├── data_examples       # examples of data produced by datasets defined in datamodule
            |   ├── epoch_metrics.html  # plots with metrics returned by module class (html)
            |   ├── epoch_metrics.jpg   # plots with metrics returned by module class (jpg)
            |   ├── epoch_metrics.yaml  # yaml with metrics
            |   ├── eval_examples       # example evaluation results (plots produced by results classes)
            |   ├── logs                # per-device logs and system monitoring metrics
            |   └── model               # model-related files (ONNX if saved, layers summary, etc.)
            └── <timestamp_2>           # resumed run timestamp (e.g. 03-22_12:10)
                ├── checkpoints
                ...
                └── model

Each training run is parametrized by [yaml config](#configs). The names shown in `<>` are defined by:
* `setup.experiment_name` define the `<experiment_name>` directory name, 
* `sertup.run_name` define the `<run_name>` directory name. If set to `null` (default), then `<run_name>` is generated automatically as `<timestamp>__<setup.dataset>_<setup.architecture>` 

For each new run there is a new results directory created (defined by current timestamp). If run is resumed (same `<run_name>` is used), then the new subrun directory (based on timestamp) is added. 

### MLFlow

By default the [mlflow](https://mlflow.org/) is used as the experiments logger (local `mlflow` server under `http://127.0.0.1:5000/` address). The runs logged in mlflow are structured a bit different than ones present in `results` directory. The main differences:
* Resuming the run is equivalent to logging to the same run (no subruns directories added),
* There is a new directory in a run artifacts called `history`, where logs and configs of each subrun are saved in their corresponding `<timestamp>` directories,
* Resuming the run overwrites previously logged `data_examples`, `logs`, `config.yaml`, `eval_examples` and `epoch_metrics` artifacts.- [Multi Person Pose Estimation with PyTorch]


## Training guide

> **_NOTE:_** Read all previous chapters before running the commands listed below.

> **_NOTE:_**  
> Adjust settings like:
> 
> * `--dataloader.batch_size` (default: 80 for hrnet, 36 for higher_hrnet)
> * `--dataloader.num_workers` (default: 4 for both tasks)
> 
> to your device capabilities


Depending on what and how you would like to train the models there exist a few possibilities (listed below). All examples assume using single GPU (to train with multiple GPUs use the `torchrun` commands from previous chapters)

### 1. Only HRNet (classifier) training

First run:
```bash
python src/classification/bin/train.py --setup.ckpt_path=None --trainer.use_DDP=False --setup.experiment_name="classification_exp" --setup.run_name="only_hrnet_run"
```

<details>
<summary>Optionally if resuming is needed:</summary>

use checkpoint from previous run:

`ckpt_path = "results/classification_exp/only_hrnet_run/<timestamp>/checkpoints/last.pt"`

```bash
python src/classification/bin/train.py --setup.ckpt_path=<ckpt_path> --trainer.use_DDP=False
```
</details>

### 2. Only HigherHRNet (keypoints) training (without pre-trained HRNet backbone)

First run:

```bash
python src/keypoints/bin/train.py --setup.ckpt_path=None --trainer.use_DDP=False --setup.experiment_name="keypoints_exp" --setup.run_name="only_higherhrnet_run"
```

<details>
<summary>Optionally if resuming is needed:</summary>

use checkpoint from previous run:

`ckpt_path = "results/keypoints_exp/only_higherhrnet_run/<timestamp>/checkpoints/last.pt"`


```bash
python src/keypoints/bin/train.py --setup.ckpt_path=<ckpt_path> --trainer.use_DDP=False --setup.experiment_name="keypoints_exp" --setup.run_name="only_higherhrnet_run"
```
</details>


### 3. Only HigherHRNet (keypoints) training (with pre-trained HRNet backbone)

> **_NOTE:_** [Downloaded](#checkpoints-with-trained-models) `hrnet_32.pt` checkpoint must be present in _pretrained_ directory.


First run:

```bash
python src/keypoints/bin/train.py --setup.ckpt_path=None --trainer.use_DDP=False --setup.experiment_name="keypoints_exp" --setup.run_name="pretrained_higherhrnet_run" --setup.pretrained_ckpt_path="pretrained/hrnet_32.pt"
```
<details>
<summary>Optionally if resuming is needed:</summary>

> **_NOTE:_** There is no need to pass the `pretrained_ckpt_path` when resuming the training since its weights were updated during training.

use checkpoint from previous run:

`ckpt_path = "results/keypoints_exp/pretrained_higherhrnet_run/<timestamp>/checkpoints/last.pt"`


```bash
python src/keypoints/bin/train.py --setup.ckpt_path=<ckpt_path> --trainer.use_DDP=False --setup.experiment_name="keypoints_exp" --setup.run_name="only_higherhrnet_run"
```
</details>


### 4. Complete, <i>"from scratch"</i> training

The complete (<i>"from scratch"</i>) training include pretraining of ClassificationHRNet and then using it as a backbone for HRNet.

1. Train classification model (HRNet backbone)
```bash
python src/classification/bin/train.py --setup.ckpt_path=None --trainer.use_DDP=False --setup.experiment_name="classification_exp" --setup.run_name="from_scratch_hrnet_pretrain_run"
```

<details>
<summary>Optionally if resuming is needed:</summary>

`ckpt_path = "results/classification_exp/from_scratch_hrnet_pretrain_run/<timestamp>/checkpoints/last.pt"`

```bash
python src/classification/bin/train.py --setup.ckpt_path=<ckpt_path> --trainer.use_DDP=False --setup.experiment_name="classification_exp" --setup.run_name="from_scratch_hrnet_pretrain_run"
```
</details>


2. Use pretrained backbone and train HigherHRNet keypoints estimation model

`pretrained_ckpt_path = "results/classification_exp/from_scratch_hrnet_pretrain_run/<timestamp>/checkpoints/last.pt"`

```bash
python src/keypoints/bin/train.py --setup.ckpt_path=None --trainer.use_DDP=False --setup.experiment_name="keypoints_exp" --setup.run_name="from_scratch_pretrained_higherhrnet_run" --setup.pretrained_ckpt_path=<pretrained_ckpt_path>
```

<details>
<summary>Optionally if resuming is needed:</summary>

`ckpt_path = "results/keypoints_exp/from_scratch_pretrained_higherhrnet_run/<timestamp>/checkpoints/last.pt"`

```bash
python src/classification/bin/train.py --setup.ckpt_path=<ckpt_path> --trainer.use_DDP=False --setup.experiment_name="classification_exp" --setup.run_name="from_scratch_pretrained_higherhrnet_run"
```
</details>