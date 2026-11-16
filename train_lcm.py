import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config import LCMConfig
from data import prepare_data
from lcm_rf import LCMRF
from networks import build_networks


def parse_args():
    parser = argparse.ArgumentParser(description="Train the LCM-RF antenna model")
    parser.add_argument("--epochs-teacher", type=int, default=None)
    parser.add_argument("--epochs-lcm", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")
    return parser.parse_args()


def to_device(batch, device):
    return tuple(tensor.to(device) for tensor in batch)


def train_teacher(model: LCMRF, loader, config: LCMConfig, epochs: int) -> list:
    optimizer = torch.optim.AdamW(
        [p for p in model.teacher.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    losses = []
    model.teacher.train()
    model.spec_encoder.train()
    for epoch in range(epochs):
        running = 0.0
        for specs, counts in loader:
            specs, counts = to_device((specs, counts), model.device)
            optimizer.zero_grad()
            loss = model.teacher_loss(specs, counts)
            loss.backward()
            optimizer.step()
            running += loss.item()
        losses.append(running / max(len(loader), 1))
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"[Teacher] epoch {epoch + 1}/{epochs}  loss={losses[-1]:.6f}", flush=True)
    model.save("lcm_teacher.pth")
    return losses


def train_student(model: LCMRF, loader, config: LCMConfig, epochs: int) -> list:
    for parameter in model.spec_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.student.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    losses = []
    model.student.train()
    for epoch in range(epochs):
        running = 0.0
        for specs, counts in loader:
            specs, counts = to_device((specs, counts), model.device)
            optimizer.zero_grad()
            loss = model.consistency_loss(specs, counts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1.0)
            optimizer.step()
            model.update_ema()
            running += loss.item()
        losses.append(running / max(len(loader), 1))
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"[LCM] epoch {epoch + 1}/{epochs}  loss={losses[-1]:.6f}", flush=True)
    model.save("lcm_rf.pth")
    return losses


def plot_losses(teacher_losses, lcm_losses, config: LCMConfig) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    if teacher_losses:
        axis.plot(teacher_losses, label="Teacher DDPM loss")
    if lcm_losses:
        axis.plot(lcm_losses, label="LCM consistency loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("LCM-RF training")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.path("lcm_training_loss.png"), dpi=120)
    plt.close(figure)


def main():
    args = parse_args()
    config = LCMConfig()
    if args.samples is not None:
        config.n_samples = args.samples
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    if args.epochs_teacher is not None:
        config.teacher_epochs = args.epochs_teacher
    if args.epochs_lcm is not None:
        config.lcm_epochs = args.epochs_lcm
    config.save()

    start = time.time()
    train_loader, _, normalizer, _ = prepare_data(config, regenerate=args.regenerate_data)
    print(f"Data ready: {len(train_loader.dataset)} training samples, {normalizer.input_min.shape[0]} specs", flush=True)

    spec_encoder, teacher, student, ema = build_networks(config)
    model = LCMRF(config, spec_encoder, student, ema, teacher)

    teacher_losses = []
    if not args.skip_teacher and config.teacher_epochs > 0:
        teacher_losses = train_teacher(model, train_loader, config, config.teacher_epochs)

    lcm_losses = []
    if config.lcm_epochs > 0:
        lcm_losses = train_student(model, train_loader, config, config.lcm_epochs)

    plot_losses(teacher_losses, lcm_losses, config)
    print(f"Training finished in {time.time() - start:.1f}s; model saved to {config.path('lcm_rf.pth')}", flush=True)


if __name__ == "__main__":
    main()
