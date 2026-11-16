import math

import torch


class SigmaSchedule:
    def __init__(self, num_steps: int, sigma_min: float, sigma_max: float):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigmas = torch.exp(torch.linspace(math.log(sigma_min), math.log(sigma_max), num_steps))

    def to(self, device) -> "SigmaSchedule":
        self.sigmas = self.sigmas.to(device)
        return self

    def index_tensor(self, indices) -> torch.Tensor:
        return self.sigmas[indices]

    def sample_indices(self, batch_size: int, low: int = 1, high: int = None) -> torch.Tensor:
        high = self.num_steps - 1 if high is None else high
        return torch.randint(low, high + 1, (batch_size,))

    @staticmethod
    def add_noise(x0: torch.Tensor, sigma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return x0 + sigma.reshape(-1, 1) * noise

    @staticmethod
    def ode_step(
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        denoiser: callable,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        estimate = denoiser(x, sigma, omega)
        return x + (sigma_next - sigma).reshape(-1, 1) * (x - estimate) / sigma.reshape(-1, 1)


def denoise_fn(network, x: torch.Tensor, sigma: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    return network.denoise(x, sigma, omega)
