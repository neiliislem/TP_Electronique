import numpy as np

from data import AntennaModel


def radiation_pattern(architecture, theta0_deg: float, n_rings: int = 5, n_theta: int = 1000) -> tuple:
    model = AntennaModel(n_rings=n_rings, n_theta=n_theta)
    af = model.array_factor(architecture, theta0_deg)
    magnitude = np.abs(af)
    normalized = magnitude / (magnitude.max() + np.finfo(float).eps)
    db = 20 * np.log10(normalized + np.finfo(float).eps)
    return model.theta_deg, np.clip(db, -40, None)


def compute_specs(architecture, theta0_deg: float, n_rings: int = 5) -> np.ndarray:
    return AntennaModel(n_rings=n_rings).specs(architecture, theta0_deg)


def compare_architectures(predicted, reference, theta0_deg: float, n_rings: int = 5) -> dict:
    pred_specs = compute_specs(predicted, theta0_deg, n_rings)
    ref_specs = compute_specs(reference, theta0_deg, n_rings)
    return {
        "predicted_specs": pred_specs,
        "reference_specs": ref_specs,
        "absolute_error": np.abs(pred_specs - ref_specs),
        "architecture_mae": float(np.mean(np.abs(np.asarray(predicted) - np.asarray(reference)))),
        "exact_match": bool(np.array_equal(np.asarray(predicted), np.asarray(reference))),
    }
