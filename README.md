# Multi Person Pose Estimation with PyTorch
[HigherHRNet](https://arxiv.org/abs/1908.10357) architecture implemented and trained from scratch using ImageNet and COCO datasets.
The model is trained in two steps:
1. Backbone pretraining on ImageNet dataset (_ClassificationHRNet_)
2. Training on COCO (_HigherHRNet_)

# Checkpoints

The checkpoints are available at [Google Drive](https://drive.google.com/drive/folders/1srwrno9YYSBy1Wqf-uDok_TpID9Mm20d?usp=sharing):

* **`hrnet_32.pt`** - backbone pretrained on the ImageNet
* **`higher_hrnet_32.pt`** - pose estimation model trained on COCO

After download place the checkpoints inside the **_pretrained_** directory.

---
# Inference

## Classification (_ClassificationHRNet_)



## Human Pose (_HigherHRNet_)

### COCO data

### Video



---

# Pretraining

## Dataset (ImageNet)

## Training

## Evaluation


---
# Training on COCO

## Dataset (COCO)

## Training

## Evaluation

**val2017**
| First Header | Second Header |
| ------------ | ------------- |
| Content Cell | Content Cell  |
| Content Cell | Content Cell  |
