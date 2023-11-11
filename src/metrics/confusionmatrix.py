"""Confusion matrix for classification tasks (i.e. segmentation)."""

from typing import Any, Callable
import numpy as np


def lazy_property(fn: Callable) -> Callable:
    """Lazily evaluated property.

    Args:
        fn (Callable): Function to decorate.

    Returns:
        Callable: Decorated function.
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self) -> Any:
        if not hasattr(self, "_recalculate") or self._recalculate is None:
            setattr(self, "_recalculate", {fn.__name__: True})
        elif (
            hasattr(self, "_recalculate")
            and fn.__name__ not in self._recalculate.keys()
        ):
            self._recalculate[fn.__name__] = True
        if self._recalculate[fn.__name__]:
            setattr(self, attr_name, fn(self))
            self._recalculate[fn.__name__] = False
        return getattr(self, attr_name)

    return _lazy_property


class ConfusionMatrix:
    """Constructs a confusion matrix for a multi-class classification problems.

    Does not support multi-label, multi-class problems.

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.

    Modified from: https://github.com/davidtvs/PyTorch-ENet
    """

    def __init__(
        self,
        num_classes: int,
        normalized: bool = False,
        ignore_index: int | None = None,
        void_label: int | None = None,
    ):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int64)
        self.normalized = normalized
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.void_label = void_label
        self.reset()

    def clone(self) -> "ConfusionMatrix":
        """Return new ConfusionMatrix object initialized same as current."""
        return ConfusionMatrix(
            num_classes=self.num_classes,
            normalized=self.normalized,
            ignore_index=self.ignore_index,
            void_label=self.void_label,
        )

    def reset(self) -> None:
        """Fill matrix with zeros"""
        self._recalculate = None
        self.conf.fill(0)

    def add(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Add values to confusion matrix.

        The shape of the confusion matrix is K x K, where K is the number
        of classes.

        Keyword arguments:
        - y_pred (numpy.ndarray): Can be an N x K tensor/array of
            y_pred scores obtained from the model for N examples and K classes,
            or an array of integer values between 0 and K-1.
        - y_true (numpy.ndarray): Can be an N x K array of
            ground-truth classes for N examples and K classes, or an array
            of integer values between 0 and K-1.

        """
        # If y_true and/or y_pred are tensors, convert them to numpy arrays
        assert (
            y_pred.shape[0] == y_true.shape[0]
        ), "num of y_true and y_pred outputs do not match"

        # convert (N, K, H, W) to (N, H, W)
        if y_pred.ndim == 4:
            y_pred = y_pred.argmax(1)  # (N, K, H, W) -> (N, H, W)
        if y_true.ndim == 4:
            y_true = y_true.argmax(1)  # (N, K, H, W) -> (N, H, W)

        # convert (N, *) to (N)
        if y_pred.ndim > 2:
            y_pred = y_pred.flatten()
        if y_true.ndim > 2:
            y_true = y_true.flatten()

        # mask out labels which are not valid (void)
        if self.void_label is not None:
            valid_labels_mask = y_true != self.void_label
            y_pred = y_pred[valid_labels_mask]
            y_true = y_true[valid_labels_mask]

        assert (
            max(y_pred) < self.num_classes and min(y_pred) >= 0
        ), "y_pred are not in [0, k-1]"
        assert (
            max(y_true) < self.num_classes and min(y_true) >= 0
        ), "y_true are not in [0, k-1]"

        # hack for bincounting 2 arrays together
        x = y_pred + self.num_classes * y_true
        bincount_2d = np.bincount(x.astype(np.int64), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        self._recalculate = None
        self.conf += conf

    @lazy_property
    def confusion_matrix(self) -> np.ndarray:
        """Confusion matrix.
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth y_trues and columns corresponds to y_pred
            y_trues.
        """
        conf = self.conf.copy()
        if self.ignore_index is not None:
            conf[:, self.ignore_index] = 0
            conf[self.ignore_index, :] = 0
        if self.normalized:
            conf = conf.astype(np.float32)
            conf /= conf.sum(1).clip(min=1e-12)[:, None]
        return conf

    @lazy_property
    def TP(self) -> np.ndarray:
        """Return True Positives."""
        return np.diag(self.confusion_matrix)

    @lazy_property
    def FP(self) -> np.ndarray:
        """Return False Positives."""
        return np.sum(self.confusion_matrix, 0) - self.TP

    @lazy_property
    def FN(self) -> np.ndarray:
        """Return False Negatives."""
        return np.sum(self.confusion_matrix, 1) - self.TP

    @lazy_property
    def iou(self) -> np.ndarray:
        """Return per class Intersection over Union (IoU)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = self.TP / (self.TP + self.FP + self.FN)
        return iou

    @lazy_property
    def mean_iou(self) -> float:
        """Return mean Intersection over Union (mIoU)."""
        return np.nanmean(self.iou).item()

    @lazy_property
    def dice_score(self) -> np.ndarray:
        """Return per class Dice Score."""
        with np.errstate(divide="ignore", invalid="ignore"):
            dice = (2 * self.TP) / (2 * self.TP + self.FP + self.FN)
        return dice

    @lazy_property
    def mean_dice_score(self) -> float:
        """Return mean Dice Score."""
        return np.nanmean(self.dice_score).item()

    def compute(self) -> dict[str, float]:
        """Return metrics useful for evaluation."""
        return {
            "mean_IoU": self.mean_iou,
            "mean_DiceScore": self.mean_dice_score,
        }
