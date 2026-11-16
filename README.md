# LCM-RF — Latent Consistency Model for Circular Antenna Array Design

Implementation of the modular architecture **"Architecture Modulaire d'un Modèle d'Antenne Circulaire par Distillation (LCM-RF)"**.

The goal is inverse antenna design: from target radiation specifications **s = [Gain, SLL, HPBW, Angle]** predict the number of elements on each of 5 concentric rings **y = [N1, N2, N3, N4, N5]**. A vector denoiser is first trained as a teacher, then distilled into a one-step (or 1–4 step) consistency model guided by the target specifications.

## Architecture

```mermaid
flowchart LR
    S["Specifications (N,4)"] --> P["1. Preprocessing & encoding<br/>s_t = (s-min)/(max-min)+1e-8"]
    P -->|s_t, t, ω| C["2. Antenna Coherence Network f_θ<br/>FiLM guidance + skip connections"]
    C -->|ŷ_0 (N,5)| L["3. LCM Scheduler<br/>1–4 steps"]
    L -->|"+ noise ε → y''_t"| C
    L --> PO["4. Post-processing<br/>denormalize + round + clip"]
    PO --> A["Predicted architecture (N,5)"]
    D["Normalized count data"] --> T["5. Teacher & PF-ODE solver<br/>y_(t+Δt) → y_t"]
    T --> E["6. EMA target + consistency loss<br/>stop-gradient, minimize L_cons"]
    E -.-> C
```

| # | Block | Implementation |
|---|-------|----------------|
| 1 | Preprocessing & encoding | Min–max normalization of specifications, `SpecEncoder` MLP producing the guidance embedding `ω` (plus a learned unconditional token) |
| 2 | Coherence network `f_θ` | VE denoiser `f(x,σ,ω) = x − σ·F_θ(x,σ,ω)` with sinusoidal σ embedding, FiLM conditioning and skip connections (`networks.py`) |
| 3 | LCM scheduler | 1–4 sampling steps; between steps the clean estimate is re-noised (the `+ bruit ε → y''_t` feedback) (`lcm_rf.py`) |
| 4 | Post-processing | Denormalization, `round()`, clipping to `[0, max_elements]` (`test_lcm.py`) |
| 5 | Teacher & PF-ODE | Denoiser trained with an MSE denoising loss; deterministic Euler step in σ used as the ODE solver (`diffusion.py`, `train_lcm.py`) |
| 6 | EMA target & loss | Exponentially averaged stop-gradient target, consistency loss with optional pseudo-Huber, plus a reconstruction anchor (`lcm_rf.py`) |

Classifier-free guidance is applied to both the student and the EMA target: `f_cfg = f_uncond + w·(f_cond − f_uncond)`.

## Installation

Python 3.10+ with:

```bash
pip install torch numpy scipy matplotlib
```

CUDA is used automatically when available (override with `--device cpu|cuda`).

## Quick start

```bash
python train_lcm.py
python test_lcm.py --steps 1
```

Training options:

```bash
python train_lcm.py --samples 1000 --epochs-teacher 60 --epochs-lcm 100 --regenerate-data
```

Inference options:

```bash
python test_lcm.py --steps 4 --index 0 --max-samples 200 --guidance 2.0
```

| Flag | Meaning |
|------|---------|
| `--steps` | Number of LCM scheduler steps (1–4) |
| `--index` | Test sample used for the detailed radiation-pattern report |
| `--max-samples` | Number of test samples used for aggregate metrics |
| `--guidance` | Override the classifier-free guidance scale |
| `--no-plot` | Skip the polar radiation-pattern figure |
| `--skip-teacher` | Skip teacher pretraining (distillation only) |

## Data

`data.py` generates the dataset at 2.45 GHz with 5 rings (radii `0.2λ + 0.5λ·r`), 0–10 elements per ring and steering angles in `[0°, 180°]`. For each architecture it computes the array factor and extracts gain, side-lobe level, HPBW and angle. Results are saved as `Minput.npy` (N,4) and `Moutput.npy` (N,5).

The split is 90/10 with seed 46. Specifications are normalized to `[0, 1]`, counts to `[-1, 1]` so that the pure-noise starting point of the consistency sampler matches the data distribution. Normalization statistics are stored in `lcm_norm.npz`.

## Configuration

All hyperparameters live in `config.py` and are saved to `lcm_config.json` at training time:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_steps` | 40 | Number of noise levels in `[sigma_min, sigma_max] = [0.002, 1]` |
| `distill_skip` | 20 | Maximum random skip between paired noise levels |
| `terminal_prob` | 0.3 | Probability of drawing the terminal (pure-noise) level |
| `ema_decay` | 0.95 | Decay of the stop-gradient target network |
| `guidance_scale` | 2.0 | Classifier-free guidance weight `w` |
| `reconstruction_weight` | 1.0 | Weight of the denoising anchor added to `L_cons` |
| `hidden_dim`, `n_blocks` | 384, 5 | Coherence network capacity |
| `teacher_epochs`, `lcm_epochs` | 60, 100 | Duration of the two training phases |
| `batch_size`, `lr` | 64, 2e-4 | Optimizer settings (AdamW) |

The saved `work_dir` is replaced by the location of `config.py` when loading, so the configuration is portable across machines.

## Results

On 100 held-out samples the trained model reaches a mean per-ring absolute error of about **3.0 elements/ring** (1 step: 3.08, 4 steps: 3.01). This is close to the dataset's information limit — a 1-nearest-neighbour baseline in specification space gives 2.91 and the global mean predictor 2.82 — because many different architectures produce nearly identical radiation patterns.

Training loss curves are saved to `lcm_training_loss.png`, predictions to `lcm_predictions.npy`, and the predicted/reference radiation patterns to `lcm_radiation_pattern.png`.

## Files

| File | Role |
|------|------|
| `config.py` | Hyperparameters and path helpers |
| `data.py` | Antenna model, dataset generation, normalization, data loaders |
| `networks.py` | Spec encoder, timestep embedding, FiLM blocks, coherence network, EMA helper |
| `diffusion.py` | Sigma schedule, noise injection, PF-ODE solver step |
| `lcm_rf.py` | LCM-RF model: guided sampling, consistency loss, checkpoint handling |
| `utils.py` | Radiation pattern and specification comparison utilities |
| `train_lcm.py` | Teacher pretraining + consistency distillation |
| `test_lcm.py` | Inference, post-processing and evaluation |
| `FFNN_training.py`, `test_model_FFNN.py`, `dataset.py`, `repartition_training_test.py`, `circular antenna array.py` | Original feed-forward baseline kept for comparison |
