import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LCMConfig


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.float().reshape(-1)
        log_sigma = torch.log(sigma.clamp_min(1e-8))
        half = self.dim // 2
        positions = torch.arange(half, device=sigma.device, dtype=torch.float32) / max(half - 1, 1)
        frequencies = torch.exp(-math.log(10000.0) * positions)
        scaled = log_sigma[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class SpecEncoder(nn.Module):
    def __init__(self, n_specs: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_specs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.null = nn.Parameter(torch.zeros(1, embed_dim))

    def forward(self, specs: torch.Tensor, conditional: torch.Tensor = None) -> torch.Tensor:
        encoded = self.mlp(specs)
        if conditional is not None:
            mask = conditional.reshape(-1, 1).to(encoded.dtype)
            encoded = mask * encoded + (1 - mask) * self.null
        return encoded

    def unconditional(self, batch_size: int, device) -> torch.Tensor:
        return self.null.expand(batch_size, -1).to(device)


class ConditioningBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        hidden = F.silu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        gamma, beta = self.film(F.silu(cond)).chunk(2, dim=-1)
        hidden = hidden * (1 + gamma) + beta
        hidden = self.dropout(self.fc2(hidden))
        return x + hidden


class CoherenceNetwork(nn.Module):
    def __init__(self, config: LCMConfig, spec_encoder: SpecEncoder = None):
        super().__init__()
        self.config = config
        self.spec_encoder = spec_encoder or SpecEncoder(config.n_specs, config.spec_embed_dim)
        self.time_embed = SinusoidalEmbedding(config.time_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(config.spec_embed_dim + config.time_embed_dim, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
        )
        self.input_proj = nn.Linear(config.n_counts + config.cond_dim, config.hidden_dim)
        self.input_skip = nn.Linear(config.n_counts, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [ConditioningBlock(config.hidden_dim, config.cond_dim, config.dropout) for _ in range(config.n_blocks)]
        )
        self.out_norm = nn.LayerNorm(config.hidden_dim)
        self.out = nn.Linear(config.hidden_dim, config.n_counts)
        self.output_skip = nn.Linear(config.n_counts, config.n_counts)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        c_in = 1.0 / torch.sqrt(1.0 + sigma.reshape(-1, 1) ** 2)
        x_in = x * c_in
        time = self.time_embed(sigma)
        cond = self.cond_proj(torch.cat([omega, time], dim=-1))
        hidden = self.input_proj(torch.cat([x_in, cond], dim=-1)) + self.input_skip(x_in)
        for block in self.blocks:
            hidden = block(hidden, cond)
        hidden = F.silu(self.out_norm(hidden))
        return self.output_skip(x_in) + self.out(hidden)

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        return x - sigma.reshape(-1, 1) * self.forward(x, sigma, omega)


def build_networks(config: LCMConfig) -> tuple:
    spec_encoder = SpecEncoder(config.n_specs, config.spec_embed_dim)
    teacher = CoherenceNetwork(config, spec_encoder)
    student = CoherenceNetwork(config, spec_encoder)
    ema = CoherenceNetwork(config, spec_encoder)
    ema.load_state_dict(student.state_dict())
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return spec_encoder, teacher, student, ema


class EMAHelper:
    def __init__(self, decay: float):
        self.decay = decay

    @torch.no_grad()
    def update(self, ema_model: nn.Module, model: nn.Module) -> None:
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for key, value in model_params.items():
            ema_params[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
        ema_buffers = dict(ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for key, value in model_buffers.items():
            ema_buffers[key].copy_(value)
