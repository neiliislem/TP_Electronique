import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, TensorDataset

from config import LCMConfig


@dataclass
class Normalizer:
    input_min: np.ndarray
    input_max: np.ndarray
    output_min: np.ndarray
    output_max: np.ndarray

    def encode_input(self, values: np.ndarray) -> np.ndarray:
        return (values - self.input_min) / (self.input_max - self.input_min + 1e-8)

    def decode_input(self, values: np.ndarray) -> np.ndarray:
        return values * (self.input_max - self.input_min + 1e-8) + self.input_min

    def encode_output(self, values: np.ndarray) -> np.ndarray:
        scaled = (values - self.output_min) / (self.output_max - self.output_min + 1e-8)
        return 2.0 * scaled - 1.0

    def decode_output(self, values: np.ndarray) -> np.ndarray:
        scaled = (values + 1.0) / 2.0
        return scaled * (self.output_max - self.output_min + 1e-8) + self.output_min

    def save(self, path: str) -> None:
        np.savez(
            path,
            input_min=self.input_min,
            input_max=self.input_max,
            output_min=self.output_min,
            output_max=self.output_max,
        )

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        with np.load(path) as data:
            return cls(
                input_min=data["input_min"],
                input_max=data["input_max"],
                output_min=data["output_min"],
                output_max=data["output_max"],
            )


class AntennaModel:
    def __init__(self, n_rings: int = 5, max_elements: int = 10, carrier_freq: float = 2.45e9, n_theta: int = 1000):
        self.n_rings = n_rings
        self.max_elements = max_elements
        self.carrier_freq = carrier_freq
        self.n_theta = n_theta
        self.wavelength = 3e8 / carrier_freq
        self.k = 2 * np.pi / self.wavelength
        self.radii = 0.2 * self.wavelength + 0.5 * self.wavelength * np.arange(n_rings)
        self.theta = np.linspace(0, 2 * np.pi, n_theta)
        self.theta_deg = np.rad2deg(self.theta)

    def array_factor(self, elements_per_ring, theta0_deg: float) -> np.ndarray:
        theta0 = np.deg2rad(theta0_deg)
        af = np.zeros_like(self.theta, dtype=complex)
        for ring, count in enumerate(elements_per_ring):
            count = int(count)
            if count == 0:
                continue
            a = self.radii[ring]
            phi_n = 2 * np.pi * np.arange(count) / count
            phase = self.k * a * (
                np.sin(self.theta)[:, None] * np.cos(phi_n)[None, :]
                - np.sin(theta0) * np.cos(phi_n)[None, :]
            )
            af += np.exp(1j * phase).sum(axis=1)
        return af

    def specs(self, elements_per_ring, theta0_deg: float) -> np.ndarray:
        af = self.array_factor(elements_per_ring, theta0_deg)
        af_abs = np.abs(af)
        gain = 20 * np.log10(af_abs.max() + np.finfo(float).eps)

        af_norm = af_abs / (af_abs.max() + np.finfo(float).eps)
        af_db = 20 * np.log10(af_norm + np.finfo(float).eps)
        af_db = np.clip(af_db, -40, None)

        max_idx = int(np.argmax(af_db))
        half_power = af_db[max_idx] - 3
        af_ext = np.concatenate([af_db, af_db, af_db])
        theta_ext = np.concatenate([self.theta_deg - 360, self.theta_deg, self.theta_deg + 360])
        max_ext = max_idx + self.n_theta
        left = np.where(af_ext[:max_ext] <= half_power)[0]
        right = np.where(af_ext[max_ext:] <= half_power)[0]
        if len(left) == 0 or len(right) == 0:
            hpbw = 180.0
        else:
            hpbw = float(theta_ext[right[0] + max_ext] - theta_ext[left[-1]])

        peaks, _ = find_peaks(af_norm, distance=5)
        if len(peaks) == 0:
            ssl = 0.0
        else:
            peak_levels = 20 * np.log10(af_norm[peaks] + np.finfo(float).eps)
            order = np.argsort(peak_levels)[::-1]
            sorted_levels = peak_levels[order]
            main = np.where(sorted_levels >= sorted_levels[0] - 1)[0]
            side = np.setdiff1d(np.arange(len(sorted_levels)), main)
            ssl = float(sorted_levels[side[0]]) if len(side) > 0 else 0.0

        return np.array([gain, ssl, hpbw, theta0_deg], dtype=np.float64)


def generate_dataset(config: LCMConfig) -> tuple:
    model = AntennaModel(n_rings=config.n_counts, max_elements=config.max_elements)
    rng = np.random.default_rng(config.seed)
    specs = np.zeros((config.n_samples, config.n_specs), dtype=np.float64)
    counts = np.zeros((config.n_samples, config.n_counts), dtype=np.float64)

    for index in range(config.n_samples):
        while True:
            architecture = rng.integers(0, config.max_elements + 1, size=config.n_counts)
            if np.count_nonzero(architecture > 0) > 0:
                break
        angle = rng.uniform(0, 180)
        specs[index] = model.specs(architecture, angle)
        counts[index] = architecture

    return specs.astype(np.float32), counts.astype(np.float32)


def load_arrays(config: LCMConfig, regenerate: bool = False) -> tuple:
    input_path = config.path("Minput.npy")
    output_path = config.path("Moutput.npy")
    if not regenerate and os.path.exists(input_path) and os.path.exists(output_path):
        specs = np.load(input_path)
        counts = np.load(output_path)
    else:
        specs, counts = generate_dataset(config)
        np.save(input_path, specs)
        np.save(output_path, counts)

    if specs.shape[0] != counts.shape[0]:
        if specs.shape[1] == counts.shape[1]:
            specs, counts = specs.T, counts.T
        elif specs.shape[1] == counts.shape[0]:
            specs = specs.T
    return specs.astype(np.float32), counts.astype(np.float32)


def prepare_data(config: LCMConfig, regenerate: bool = False) -> tuple:
    specs, counts = load_arrays(config, regenerate)
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(specs))
    n_test = int(round(config.test_ratio * len(specs)))
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    train_specs, train_counts = specs[train_idx], counts[train_idx]
    test_specs, test_counts = specs[test_idx], counts[test_idx]

    normalizer = Normalizer(
        input_min=train_specs.min(axis=0),
        input_max=train_specs.max(axis=0),
        output_min=train_counts.min(axis=0),
        output_max=train_counts.max(axis=0),
    )
    normalizer.save(config.path("lcm_norm.npz"))

    train_loader = make_loader(
        normalizer.encode_input(train_specs),
        normalizer.encode_output(train_counts),
        config.batch_size,
        shuffle=True,
    )
    test_loader = make_loader(
        normalizer.encode_input(test_specs),
        normalizer.encode_output(test_counts),
        config.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader, normalizer, (test_specs, test_counts)


def make_loader(specs: np.ndarray, counts: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(specs).float(), torch.from_numpy(counts).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
