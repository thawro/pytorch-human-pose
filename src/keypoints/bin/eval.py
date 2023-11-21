import torch
from tqdm.auto import tqdm

from src.utils.config import ROOT
from src.utils.model import seed_everything
from src.keypoints.visualization import plot_heatmaps
from src.keypoints.bin.utils import create_datamodule, create_model
from src.keypoints.bin.config import cfg
from src.keypoints.metrics import KeypointsMetrics

EVAL_RUN_NAME = "18-11-2023_17:25:54_train_MPII_LR(0.001)"
cfg.setup.ckpt_path = str(ROOT / f"results/test/{EVAL_RUN_NAME}/checkpoints/last.pt")

eps = 1e-8


"""Train the model"""


def main() -> None:
    seed_everything(cfg.setup.seed)
    torch.set_float32_matmul_precision("medium")

    datamodule = create_datamodule(cfg)
    val_dl = datamodule.val_dataloader
    model = create_model(cfg)
    metrics = KeypointsMetrics()

    ckpt = torch.load(cfg.setup.ckpt_path)["module"]["model"]
    model.load_state_dict(ckpt)
    model.to(cfg.setup.device)
    avg_accuracies = []
    for batch in tqdm(val_dl):
        images, stages_target_heatmaps, target_weights = batch
        images = images.to(cfg.setup.device)

        with torch.no_grad():
            stages_pred_heatmaps = model(images)

        inv_processing = datamodule.transform.inverse_preprocessing

        numpy_images = inv_processing(images.detach().cpu().numpy())
        h, w = numpy_images.shape[:2]

        pred_heatmaps = stages_pred_heatmaps[-1]
        target_heatmaps = stages_target_heatmaps[-1].to(cfg.setup.device)

        m = metrics.calculate_metrics(pred_heatmaps, target_heatmaps)

        avg_accuracies.append(m["PCK"])
    print(avg_accuracies)
    print(sum(avg_accuracies) / len(avg_accuracies))


if __name__ == "__main__":
    main()
