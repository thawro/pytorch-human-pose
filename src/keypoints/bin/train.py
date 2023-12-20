"""Train the model"""

import torch
from src.logging import TerminalLogger, get_pylogger
from src.base.trainer import Trainer
from src.utils.model import seed_everything

from src.keypoints.bin.utils import create_datamodule, create_callbacks, create_module
from src.keypoints.bin.config import cfg

log = get_pylogger(__name__)


def main() -> None:
    seed_everything(cfg.setup.seed)
    torch.set_float32_matmul_precision("medium")

    datamodule = create_datamodule(cfg)
    labels = datamodule.train_ds.labels
    module = create_module(cfg, labels)

    logger = TerminalLogger(cfg.logs_path, config=cfg.to_dict())
    callbacks = create_callbacks(cfg)

    trainer = Trainer(
        logger=logger,
        device=cfg.setup.device,
        callbacks=callbacks,
        max_epochs=cfg.setup.max_epochs,
        limit_batches=cfg.setup.limit_batches,
        log_every_n_steps=cfg.setup.log_every_n_steps,
    )
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()


# TODO: add lr scheduler
# TODO: add halfbody augmentation
# TODO: create training schemes same as in articles for each approach
# TODO: zrobic osobna klase datasetu per dataset (MPII, COCO) i w nich zaimplementowac metryki do ewaluacji:
# MPII: PCKh@0.5
# COCO: OKS
# TODO: dodac do inferencji model pytorchowy
# TODO: dodac do inferecji dockera
# TODO: zrobic apke (gradio?), ktora bedzie korzystac z dockera
# TODO: odpalic MPPE
# TODO: logowanie metryk nie w liscie, tylko w slowniku, zeby mozna bylo sprecyzowac krok logowania


# TODO!!!: wyliczanie PCKh@0.5 i OKS dla wersji SPPE i MPPE
#  PCKh: [X] SPPE   [ ] MPPE
#   OKS: [ ] SPPE   [ ] MPPE

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
