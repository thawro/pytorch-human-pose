import torch.distributed as dist
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, use_distributed: bool):
        self.name = name
        self.use_distributed = use_distributed
        self.reset()

    def new(self) -> "AverageMeter":
        return AverageMeter(self.name, self.use_distributed)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        if self.use_distributed:
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        if self.count == 0:
            return
        self.avg = self.sum / self.count

    def __str__(self):
        return self.summary()

    def summary(self):
        return f"{self.name}: {self.avg:.3f}"


class Meters:
    def __init__(
        self,
        meters: dict[str, AverageMeter] | None = None,
        use_distributed: bool = True,
    ):
        if meters is None:
            meters = {}
        self.meters = meters
        self.use_distributed = use_distributed

    def reset(self):
        for name in self.meters:
            self.meters[name].reset()

    def all_reduce(self):
        for name in self.meters:
            self.meters[name].all_reduce()

    def update(self, metrics: dict[str, float], n: int):
        for name, value in metrics.items():
            if name not in self.meters:
                # create new meter if metric not yet logged
                self.meters[name] = AverageMeter(name, self.use_distributed)
            self.meters[name].update(value, n=n)

    def __str__(self):
        txt = ""
        for name in self.meters:
            txt += self.meters[name].summary() + "\n"
        return txt
