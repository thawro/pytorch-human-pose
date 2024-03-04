import torchvision.transforms as T


class ClassificationTransform:
    def __init__(
        self,
        out_size: tuple[int, int] = (224, 224),
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        self.train = T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(out_size[0], antialias=True),
                T.RandomHorizontalFlip(),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.inference = T.Compose(
            [
                T.ToTensor(),
                T.Resize(int(out_size[0] / 0.875), antialias=True),
                T.CenterCrop(out_size[0]),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.out_size = out_size
