import torchvision.transforms as T

from src.base.transforms.base import ImageTransform


class ClassificationTransform(ImageTransform):
    def __init__(
        self,
        out_size: tuple[int, int] = (224, 224),
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        super().__init__(out_size, mean, std)
        self.train = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(out_size[0], antialias=True),
                T.RandomHorizontalFlip(),
                self.normalize,
            ]
        )

        self.inference = T.Compose(
            [
                T.ToTensor(),
                T.Resize(int(out_size[0] / 0.875), antialias=True),
                T.CenterCrop(out_size[0]),
                self.normalize,
            ]
        )
        self.out_size = out_size
