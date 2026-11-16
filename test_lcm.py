import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import LCMConfig
from data import prepare_data
from lcm_rf import LCMRF
from utils import compare_architectures, radiation_pattern


def parse_args():
    parser = argparse.ArgumentParser(description="Inference and evaluation for LCM-RF")
    parser.add_argument("--steps", type=int, default=1, help="Number of LCM scheduler steps (1-4)")
    parser.add_argument("--index", type=int, default=0, help="Test sample index to inspect")
    parser.add_argument("--max-samples", type=int, default=200, help="Samples used for aggregate metrics")
    parser.add_argument("--guidance", type=float, default=None, help="Override classifier-free guidance scale")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def postprocess(normalized: np.ndarray, normalizer, max_elements: int) -> np.ndarray:
    decoded = normalizer.decode_output(normalized)
    return np.clip(np.round(decoded), 0, max_elements).astype(int)


def main():
    args = parse_args()
    config = LCMConfig.load()
    _, _, normalizer, (test_specs, test_counts) = prepare_data(config)

    model = LCMRF.load(config)
    model.eval_mode()

    specs_tensor = torch.from_numpy(normalizer.encode_input(test_specs[:args.max_samples])).float().to(model.device)
    denorm = normalizer.decode_input(specs_tensor.cpu().numpy())
    predicted_norm = model.sample(specs_tensor, steps=args.steps, guidance=args.guidance).cpu().numpy()
    predicted = postprocess(predicted_norm, normalizer, config.max_elements)
    reference = test_counts[:len(predicted)].astype(int)

    np.save(config.path("lcm_predictions.npy"), predicted)

    mae = np.mean(np.abs(predicted - reference), axis=0)
    exact = np.mean(np.all(predicted == reference, axis=1))
    print(f"\nInference: {args.steps} LCM step(s), {len(predicted)} samples")
    print(f"Architecture MAE per ring: {np.round(mae, 3)}")
    print(f"Mean architecture MAE: {mae.mean():.3f} elements/ring")
    print(f"Exact match rate: {exact * 100:.2f}%")

    index = min(args.index, len(predicted) - 1)
    theta0 = float(denorm[index, 3])
    report = compare_architectures(predicted[index], reference[index], theta0, config.n_counts)
    print(f"\nSample #{index}: theta0 = {theta0:.2f} deg")
    print(f"  reference   : {reference[index]}")
    print(f"  predicted   : {predicted[index]}")
    print(f"  ref specs   : {np.round(report['reference_specs'], 2)}")
    print(f"  pred specs  : {np.round(report['predicted_specs'], 2)}")
    print(f"  spec |err|  : {np.round(report['absolute_error'], 2)}")
    print(f"  arch MAE    : {report['architecture_mae']:.2f}")

    if not args.no_plot:
        theta, ref_db = radiation_pattern(reference[index], theta0, config.n_counts)
        _, pred_db = radiation_pattern(predicted[index], theta0, config.n_counts)
        figure, axis = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
        axis.plot(np.deg2rad(theta), ref_db, label="Reference", linewidth=2)
        axis.plot(np.deg2rad(theta), pred_db, label="Predicted", linewidth=2, linestyle="--")
        axis.set_rlim(-40, 0)
        axis.set_title("LCM-RF radiation pattern")
        axis.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
        figure.tight_layout()
        figure.savefig(config.path("lcm_radiation_pattern.png"), dpi=120)
        plt.close(figure)
        print(f"Plots saved in {config.work_dir}")


if __name__ == "__main__":
    main()
