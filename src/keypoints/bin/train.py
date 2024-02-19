"""Train the model"""

from src.base.bin.train import train

from src.utils.config import RESULTS_PATH, YAML_EXP_PATH

from src.keypoints.config import KeypointsConfig
from src.utils.files import load_yaml


def main() -> None:
    cfg_path = YAML_EXP_PATH / "keypoints" / "higher_hrnet_32.yaml"
    cfg = load_yaml(cfg_path)

    pretrained_ckpt_path = "/home/thawro/Desktop/projects/pytorch-human-pose/results/classification/02-15_10:12___imagenet_HRNet/02-19_09:14/checkpoints/best.pt"
	
    ckpt_path = f"{str(RESULTS_PATH)}/keypoints/01-23_17:59___MPPE_COCO_OriginalHigherHRNet/01-25_08:32/checkpoints/last.pt"
    ckpt_path = None

    cfg["setup"]["ckpt_path"] = ckpt_path
    cfg["setup"]["pretrained_ckpt_path"] = pretrained_ckpt_path
    cfg["trainer"]["limit_batches"] = -1
    cfg["trainer"]["use_distributed"] = True

    cfg = KeypointsConfig.from_dict(cfg)

    train(cfg)


if __name__ == "__main__":
    main()


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
