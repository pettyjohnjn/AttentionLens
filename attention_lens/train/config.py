from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class TrainConfig:
    # Training-specific arguments
    lr: float = field(default=1e-2)
    max_epochs: int = field(default=10)
    max_checkpoint_num: int = field(default=10)
    num_nodes: int = field(default=1)
    mixed_precision: bool = field(default=True)
    checkpoint_mode: str = field(default="step")
    num_steps_per_checkpoint: int = field(default=5)
    accumulate_grad_batches: int = field(default=10)
    stopping_delta: float = field(default=1e-7)
    stopping_patience: int = field(default=2)
    reload_checkpoint: Optional[Path | str] = field(default=None)
    strategy: str = field(default="deepspeed_stage_2")
    checkpoint_dir: Path = field(default=Path("checkpoint"))
    # Lens-specific arguments
    model_name: str = field(default="gpt2")
    lora_rank: int = field(default=8)
    # Data module-specific arguments
    data_dir: Path | str = field(default="/grand/SuperBERT/pettyjohnjn/cache/datasets/monology___pile-uncopyrighted/default/0.0.0")
    split: str = field(default="train")
    batch_size: int = field(default=1)
    data_num_workers: int = field(default=1)
    data_pin_memory: bool = field(default=True)
    chunk_size: int = field(default=256)
    chunk: bool = field(default=False)


    def __post_init__(self):
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.reload_checkpoint, str):
            self.checkpoint_dir = Path(self.reload_checkpoint)