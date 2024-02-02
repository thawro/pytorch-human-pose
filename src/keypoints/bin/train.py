"""Train the model"""

import torch

from src.logging import TerminalLogger
from src.base.trainer import Trainer
from src.utils.model import seed_everything

from src.keypoints.bin.utils import (
    create_datamodule,
    create_callbacks,
    create_module,
    create_model,
)
from src.keypoints.bin.config import create_config, EXPERIMENT_NAME

from torch.distributed import init_process_group, destroy_process_group
import os
from src.utils.config import RESULTS_PATH
import torch.backends.cudnn as cudnn


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main(dataset, mode, arch, ckpt_path) -> None:
    ddp_setup()

    # if fp16_enabled:
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    assert (
        torch.backends.cudnn.enabled
    ), "fp16 mode requires cudnn backend to be enabled."

    rank = int(os.environ["LOCAL_RANK"])

    cfg = create_config(dataset, mode, arch, device_id=rank, ckpt_path=ckpt_path)

    seed_everything(cfg.setup.seed)
    torch.set_float32_matmul_precision("medium")

    datamodule = create_datamodule(cfg)
    labels = datamodule.train_ds.labels

    model = create_model(cfg)

    module = create_module(cfg, model, labels)

    logger = TerminalLogger(cfg.logs_path, config=cfg.to_dict())
    callbacks = create_callbacks(cfg)

    trainer = Trainer(logger=logger, callbacks=callbacks, **cfg.trainer.to_dict())
    trainer.fit(module, datamodule, ckpt_path=cfg.setup.ckpt_path)
    destroy_process_group()


if __name__ == "__main__":
    MODE = "MPPE"
    DATASET = "COCO"
    ARCH = "HigherHRNet"
    ARCH = "OriginalHigherHRNet"
    RUN_NAME = "01-23_17:59"
    RUN_SUBDIR = "01-24_07:37"
    PREFIX = ""
    EXP_RUN_NAME = f"{RUN_NAME}__{PREFIX}_{MODE}_{DATASET}_{ARCH}"
    EXPERIMENT_NAME = "test"
    CKPT_PATH = f"{str(RESULTS_PATH)}/{EXPERIMENT_NAME}/{EXP_RUN_NAME}/{RUN_SUBDIR}/checkpoints/last.pt"
    # CKPT_PATH = None

    main(DATASET, MODE, ARCH, CKPT_PATH)


# TODO: add halfbody augmentation
# TODO: create training schemes same as in articles for each approach

# TODO: dodac do inferencji model pytorchowy
# TODO: dodac do inferecji dockera
# TODO: zrobic apke (gradio?), ktora bedzie korzystac z dockera

# TODO: ewaluacja SPPE stosujac detektor obiektow (dla COCO wtedy uzyc cocoapi)
# TODO: sprawdzic COCO val split (dziwnie ciezkie przypadki tam sa)
# TODO: dodac te transformy z wycinaniem losowych kwadracikow
# TODO: pretrain on the imagenet

# TODO: dodac pin memory na heatmapach, keypointsach i visibilities
# TODO: zrobic init sieci w mojej implementacji tak jak w paperze
# TODO: zmodyfikowac moja implementacje, tak zeby tagi sie liczyly tylko dla pierwszego staga

# TODO: uzywac original hrneta pretrenowanego na imagenecie lub pretrenowac swojego
# TODO: bez sigmoidy -> lepiej (ale musi byc pretrenowane)
# TODO: uwazac na resume -> pozmieniane parametry
"""
Hourglass:
	1:1 aspect ratio
	256x256 wycentrowane
	rotation (+/- 30 degrees), and scaling (.75-1.25)
	RMSProp
	lr: 2.5e-4 do wysycenia, potem 5e-5
	flip heatmap -> agregacja
	1px gauss
	quarter px offset
	MPII: PCKh
	
SimpleBaseline:
	4:3 aspect ratio
	256x192 wycentrowane
	rotation (+/- 40 degrees), scaling (0.7-1.3) and flip
	lr: 1e-3, 1e-4 (90epoka), 1e-5 (120 epoka) (lacznie 140 epok)
	Adam
	batch_size: 128
	flip heatmap -> agregacja
	quarter px offset
	COCO: OKS metric
	2px gauss
	
HRNet:
	4:3 aspect ratio
	256x192 wycentrowane
	rotation (+/- 45 degrees), scaling (0.65-1.35) and flip
	lr: 1e-3, 1e-4 (170 epoka), 1e-5 (200 epoka) (lacznie 210 epok)
	Adam
	batch_size: 128
	flip heatmap -> agregacja
	quarter px offset
	COCO: OKS metric
	MPII: PCKh@0.5
	1px gauss
	
"""
