import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LCMConfig
from diffusion import SigmaSchedule
from networks import CoherenceNetwork, EMAHelper, SpecEncoder


class LCMRF:
    def __init__(self, config: LCMConfig, spec_encoder: SpecEncoder, student: CoherenceNetwork,
                 ema: CoherenceNetwork, teacher: CoherenceNetwork = None):
        self.config = config
        self.device = config.resolve_device()
        self.spec_encoder = spec_encoder.to(self.device)
        self.student = student.to(self.device)
        self.ema = ema.to(self.device)
        self.teacher = teacher.to(self.device) if teacher is not None else None
        self.schedule = SigmaSchedule(config.num_steps, config.sigma_min, config.sigma_max).to(self.device)
        self.ema_helper = EMAHelper(config.ema_decay)

    def train_mode(self) -> None:
        self.spec_encoder.train()
        self.student.train()
        if self.teacher is not None:
            self.teacher.eval()

    def eval_mode(self) -> None:
        self.spec_encoder.eval()
        self.student.eval()
        self.ema.eval()

    def parameters(self):
        return [p for p in self.student.parameters() if p.requires_grad]

    def _omega(self, specs: torch.Tensor, conditional: torch.Tensor = None) -> torch.Tensor:
        return self.spec_encoder(specs, conditional)

    def _guided(self, network: nn.Module, x: torch.Tensor, sigma: torch.Tensor,
                omega: torch.Tensor, guidance: float) -> torch.Tensor:
        if abs(guidance - 1.0) < 1e-6:
            return network.denoise(x, sigma, omega)
        uncond = self.spec_encoder.unconditional(x.shape[0], x.device)
        pred_uncond = network.denoise(x, sigma, uncond)
        pred_cond = network.denoise(x, sigma, omega)
        return pred_uncond + guidance * (pred_cond - pred_uncond)

    def teacher_loss(self, specs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        omega = self._omega(specs)
        index = self.schedule.sample_indices(specs.shape[0])
        sigma = self.schedule.index_tensor(index)
        noise = torch.randn_like(counts)
        x = self.schedule.add_noise(counts, sigma, noise)
        estimate = self.teacher.denoise(x, sigma, omega)
        return F.mse_loss(estimate, counts)

    def consistency_loss(self, specs: torch.Tensor, counts: torch.Tensor,
                         guidance: float = None, skip: int = None) -> torch.Tensor:
        guidance = self.config.guidance_scale if guidance is None else guidance
        skip = self.config.distill_skip if skip is None else skip
        batch_size = specs.shape[0]
        omega = self._omega(specs)

        index = self.schedule.sample_indices(batch_size, low=1)
        if self.config.terminal_prob > 0:
            terminal = torch.rand(batch_size, device=index.device) < self.config.terminal_prob
            index = torch.where(terminal, self.schedule.num_steps - 1, index)
        sigma = self.schedule.index_tensor(index)
        if skip > 1:
            skips = torch.randint(1, skip + 1, (batch_size,), device=index.device)
        else:
            skips = torch.full_like(index, skip)
        prev_index = (index - skips).clamp(min=0)
        sigma_prev = self.schedule.index_tensor(prev_index)

        noise = torch.randn_like(counts)
        x = self.schedule.add_noise(counts, sigma, noise)

        with torch.no_grad():
            x_pseudo = self.schedule.ode_step(
                x, sigma, sigma_prev,
                lambda noisy, sig, cond: self.teacher.denoise(noisy, sig, cond),
                omega,
            )
            target = self._guided(self.ema, x_pseudo, sigma_prev, omega, guidance)

        prediction = self._guided(self.student, x, sigma, omega, guidance)
        loss = self._loss(prediction, target)
        if self.config.reconstruction_weight > 0:
            reconstruction = self.student.denoise(x, sigma, omega)
            loss = loss + self.config.reconstruction_weight * F.mse_loss(reconstruction, counts)
        return loss

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = self.config.huber_c
        if c <= 0:
            return F.mse_loss(prediction, target)
        return (torch.sqrt((prediction - target) ** 2 + c ** 2) - c).mean()

    @torch.no_grad()
    def sample(self, specs: torch.Tensor, steps: int = None, guidance: float = None,
               seed: int = None) -> torch.Tensor:
        steps = self.config.inference_steps if steps is None else steps
        guidance = self.config.guidance_scale if guidance is None else guidance
        batch_size = specs.shape[0]
        n_counts = self.config.n_counts
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        total = self.schedule.num_steps - 1
        indices = torch.linspace(0, total, steps + 1).round().long().flip(0).to(self.device)
        sigma_max = self.schedule.sigmas[total]
        noise = torch.randn((batch_size, n_counts), device=self.device, generator=generator)
        x = sigma_max * noise
        omega = self._omega(specs)

        prediction = None
        for step in range(steps):
            sigma = self.schedule.sigmas[indices[step]].expand(batch_size)
            prediction = self._guided(self.student, x, sigma, omega, guidance)
            if step < steps - 1:
                sigma_next = self.schedule.sigmas[indices[step + 1]].expand(batch_size)
                noise_step = torch.randn(prediction.shape, device=prediction.device, generator=generator)
                x = prediction + sigma_next.reshape(-1, 1) * noise_step
        return prediction

    def update_ema(self) -> None:
        self.ema_helper.update(self.ema, self.student)

    def save(self, name: str = "lcm_rf.pth") -> None:
        torch.save({
            "config": self.config.to_dict(),
            "spec_encoder": self.spec_encoder.state_dict(),
            "student": self.student.state_dict(),
            "ema": self.ema.state_dict(),
            "teacher": self.teacher.state_dict() if self.teacher is not None else None,
        }, self.config.path(name))

    @classmethod
    def load(cls, config: LCMConfig, name: str = "lcm_rf.pth") -> "LCMRF":
        spec_encoder = SpecEncoder(config.n_specs, config.spec_embed_dim)
        student = CoherenceNetwork(config, spec_encoder)
        ema = CoherenceNetwork(config, spec_encoder)
        teacher = CoherenceNetwork(config, spec_encoder)
        model = cls(config, spec_encoder, student, ema, teacher)
        checkpoint = torch.load(config.path(name), map_location=model.device, weights_only=False)
        model.spec_encoder.load_state_dict(checkpoint["spec_encoder"])
        model.student.load_state_dict(checkpoint["student"])
        model.ema.load_state_dict(checkpoint["ema"])
        if checkpoint.get("teacher") is not None:
            model.teacher.load_state_dict(checkpoint["teacher"])
        return model


