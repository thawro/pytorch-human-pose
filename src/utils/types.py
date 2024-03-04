from typing import Literal

_stage = Literal["train", "val", "eval_val", "sanity"]
_accelerator = Literal["cpu", "gpu"]
_split = Literal["train", "val", "test"]
