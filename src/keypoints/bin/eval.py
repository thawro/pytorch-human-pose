import torch
from tqdm.auto import tqdm

from src.utils.model import seed_everything
from src.keypoints.visualization import plot_heatmaps
from src.keypoints.results import KeypointsResult
from src.keypoints.bin.utils import create_datamodule, create_model
from src.keypoints.bin.config import cfg


cfg.setup.ckpt_path = "/home/shate/Desktop/projects/pytorch-human-pose/results/test/16-11-2023_21:15:04_train_MPII_LR(0.001)/checkpoints/last.pt"

eps = 1e-8


def get_preds(scores):
    """get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, "Score maps should be 4-dim"
    batch_size, num_kpts, h, w = scores.shape
    flat_scores = scores.view(batch_size, num_kpts, -1)
    idxs = torch.argmax(flat_scores, 2)
    idxs = idxs.view(batch_size, num_kpts, 1) + 1
    preds = idxs.repeat(1, 1, 2).float()
    preds[:, :, 0] = (preds[:, :, 0] - 1) % w + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / w) + 1
    return preds


def calc_dists(preds, target, normalize):
    sqared_diff = (preds - target) ** 2
    dists = torch.sum(sqared_diff, dim=-1) ** (1 / 2)
    dists = dists / normalize.unsqueeze(-1)
    return dists


def get_kpt_accuracy(dist, thr=0.5):
    """Return percentage below threshold while ignoring values with a -1"""
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1


def accuracy(output, target, idxs=None, thr=0.5):
    """Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations"""
    batch_size, num_joints, h, w = output.shape
    if idxs is None:
        idxs = list(range(num_joints))

    target_mask = target.sum(dim=(2, 3)) > 0
    preds = get_preds(output)
    gts = get_preds(target)
    norm = torch.ones(batch_size, device=preds.device) * w / 10

    dists = calc_dists(preds, gts, norm)

    dists[~target_mask] = -1

    acc = torch.zeros(num_joints)
    avg_acc = 0
    count = 0

    for i in range(num_joints):
        kpt_acc = get_kpt_accuracy(dists[:, idxs[i]], thr)
        acc[i] = kpt_acc
        if kpt_acc >= 0:
            avg_acc = avg_acc + kpt_acc
            count += 1

    if count != 0:
        avg_acc /= count
    return avg_acc, acc


"""Train the model"""


def main() -> None:
    seed_everything(cfg.setup.seed)
    torch.set_float32_matmul_precision("medium")

    datamodule = create_datamodule(cfg)
    val_dl = datamodule.val_dataloader
    model = create_model(cfg)

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

        avg_acc, acc = accuracy(pred_heatmaps, target_heatmaps)
        avg_accuracies.append(avg_acc)
    print(avg_accuracies)
    print(sum(avg_accuracies) / len(avg_accuracies))


if __name__ == "__main__":
    main()
