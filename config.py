from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class LCMConfig:
    n_specs: int = 4
    n_counts: int = 5
    n_samples: int = 1000
    max_elements: int = 10
    test_ratio: float = 0.10
    seed: int = 46

    spec_embed_dim: int = 64
    time_embed_dim: int = 64
    cond_dim: int = 64
    hidden_dim: int = 384
    n_blocks: int = 5
    dropout: float = 0.0

    num_steps: int = 40
    sigma_min: float = 0.002
    sigma_max: float = 1.0

    distill_skip: int = 20
    ema_decay: float = 0.95
    guidance_scale: float = 2.0
    uncond_prob: float = 0.0
    huber_c: float = 0.0
    reconstruction_weight: float = 1.0
    terminal_prob: float = 0.3

    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-4
    teacher_epochs: int = 60
    lcm_epochs: int = 100
    log_every: int = 50

    inference_steps: int = 1
    device: str = "auto"
    work_dir: str = field(default_factory=lambda: str(Path(__file__).resolve().parent))

    def path(self, name: str) -> str:
        return str(Path(self.work_dir) / name)

    def to_dict(self):
        return asdict(self)

    def save(self, name: str = "lcm_config.json") -> None:
        with open(self.path(name), "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, name: str = "lcm_config.json") -> "LCMConfig":
        path = Path(__file__).resolve().parent / name
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def resolve_device(self):
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
