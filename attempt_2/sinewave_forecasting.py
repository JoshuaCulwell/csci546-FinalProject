import os
import json
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt

from training_utils import (
    build_model_from_config,
    sensitivity_analysis,
    randomized_search,
)

# -----------------------------
# Repro + device
# -----------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Sinewave forecasting dataset
# -----------------------------

class SineWaveDataset(Dataset):
    """
    Each sample:
      input:  [seq_len, 1]  (first seq_len points)
      target: [seq_len, 1]  (next seq_len points)
    """

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        freq_range=(0.5, 2.0),
        amp_range=(0.5, 1.5),
        phase_range=(0.0, 2 * math.pi),
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.noise_std = noise_std

        self.inputs, self.targets = self._generate()

    def _generate(self):
        xs = []
        ys = []
        t = torch.linspace(0, 1, self.seq_len + 1)  # +1 for forecasting

        for _ in range(self.num_samples):
            freq = random.uniform(*self.freq_range)
            amp = random.uniform(*self.amp_range)
            phase = random.uniform(*self.phase_range)

            signal = amp * torch.sin(2 * math.pi * freq * t + phase)
            #signal += self.noise_std * torch.randn_like(signal)

            x = signal[:-1].unsqueeze(-1)  # [seq_len, 1]
            y = signal[1:].unsqueeze(-1)   # [seq_len, 1]

            xs.append(x)
            ys.append(y)

        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_sine_dataloaders(
    num_samples=3000,
    seq_len=50,
    batch_size=64,
    train_frac=0.7,
    val_frac=0.15,
):
    dataset = SineWaveDataset(num_samples=num_samples, seq_len=seq_len)

    n_train = int(train_frac * num_samples)
    n_val = int(val_frac * num_samples)
    n_test = num_samples - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# -----------------------------
# Loss + metric for regression
# -----------------------------

loss_fn = nn.MSELoss()


def metric_fn(pred, target):
    # Higher is better: negative MSE
    return -nn.functional.mse_loss(pred, target).item()




# -----------------------------
# Final training loop (50 epochs)
# -----------------------------

def train_final_model(
    config,
    train_loader,
    val_loader,
    device,
    num_epochs,
    checkpoints_dir,
):
    os.makedirs(checkpoints_dir, exist_ok=True)

    model = build_model_from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metric": [],
    }

    best_val_metric = -math.inf
    best_epoch = -1

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_count += batch_size

        avg_train_loss = total_train_loss / max(total_train_count, 1)

        # ---- Val ----
        model.eval()
        total_val_loss = 0.0
        total_val_metric = 0.0
        total_val_count = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                metric = metric_fn(logits, y)

                batch_size = x.size(0)
                total_val_loss += loss.item() * batch_size
                total_val_metric += metric * batch_size
                total_val_count += batch_size

        avg_val_loss = total_val_loss / max(total_val_count, 1)
        avg_val_metric = total_val_metric / max(total_val_count, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_metric"].append(avg_val_metric)

        # Save epoch checkpoint
        epoch_ckpt_path = os.path.join(
            checkpoints_dir, f"epoch_{epoch + 1:03d}.pt"
        )
        torch.save(model.state_dict(), epoch_ckpt_path)

        # Track best model by validation metric
        if avg_val_metric > best_val_metric:
            best_val_metric = avg_val_metric
            best_epoch = epoch + 1
            best_path = os.path.join(checkpoints_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)

        print(
            f"[Final Train] Epoch {epoch+1}/{num_epochs} "
            f"TrainLoss={avg_train_loss:.4f} "
            f"ValLoss={avg_val_loss:.4f} "
            f"ValMetric={avg_val_metric:.4f}"
        )

    return history, best_epoch, best_val_metric


# -----------------------------
# Metrics + plotting helpers
# -----------------------------

def compute_test_metrics(model, test_loader, device):
    model.eval()
    mse_list = []
    mae_list = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            mse_list.append(nn.functional.mse_loss(pred, y).item())
            mae_list.append(nn.functional.l1_loss(pred, y).item())

    test_mse = float(sum(mse_list) / len(mse_list))
    test_mae = float(sum(mae_list) / len(mae_list))
    return test_mse, test_mae


def compute_horizon_errors(model, test_loader, device, horizons):
    model.eval()
    horizon_mse = {h: [] for h in horizons}

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)  # [B, T, 1]
            B, T, _ = pred.shape

            for h in horizons:
                if h <= T:
                    p = pred[:, h - 1, :]
                    t = y[:, h - 1, :]
                    mse = nn.functional.mse_loss(p, t).item()
                    horizon_mse[h].append(mse)

    horizon_mse = {h: float(sum(v) / len(v)) for h, v in horizon_mse.items()}
    return horizon_mse


def compute_spectral_error(model, test_loader, device):
    model.eval()
    spectral_errors = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)  # [B, T, 1]

            # FFT along time dimension
            y_fft = torch.fft.rfft(y.squeeze(-1), dim=1)
            p_fft = torch.fft.rfft(pred.squeeze(-1), dim=1)

            y_mag = torch.abs(y_fft)
            p_mag = torch.abs(p_fft)

            mse = nn.functional.mse_loss(p_mag, y_mag).item()
            spectral_errors.append(mse)

    return float(sum(spectral_errors) / len(spectral_errors))


def plot_loss_curves(history, title, save_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_example_forecast(model, test_loader, device, title, save_path):
    model.eval()
    x_ex, y_ex = next(iter(test_loader))
    x_ex = x_ex.to(device)
    y_ex = y_ex.to(device)

    with torch.no_grad():
        pred_ex = model(x_ex)

    x0 = x_ex[0].cpu().numpy().squeeze(-1)
    y0 = y_ex[0].cpu().numpy().squeeze(-1)
    p0 = pred_ex[0].cpu().numpy().squeeze(-1)

    plt.figure()
    plt.plot(range(len(x0)), x0, label="Input")
    plt.plot(range(1, len(y0) + 1), y0, label="Target (Future)")
    plt.plot(range(1, len(p0) + 1), p0, label="Prediction")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_horizon_errors(horizon_mse, title, save_path):
    horizons = sorted(horizon_mse.keys())
    values = [horizon_mse[h] for h in horizons]

    plt.figure()
    plt.plot(horizons, values, marker="o")
    plt.xlabel("Horizon (steps)")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -----------------------------
# Full experiment per transition
# -----------------------------

def run_experiment_for_transition(
    transition_type: str,
    train_loader,
    val_loader,
    test_loader,
    results_root: str,
    sa_epochs: int = 3,
    rs_epochs: int = 10,
    rs_trials: int = 8,
    final_epochs: int = 100,
):
    os.makedirs(results_root, exist_ok=True)

    # ---- Base config ----
    base_config = {
        "input_dim": 1,
        "output_dim": 1,
        "hidden_dim": 32,
        "transition_type": transition_type,
        "learning_rate": 1e-3,
        # MLP transition defaults
        "width": 64,
        "depth": 2,
        # Output MLP
        "out_width": 64,
        "out_depth": 2,
        # Diffusion defaults (ignored for MLP)
        "num_diffusion_steps": 4,
        "denoising_hidden_dim": 64,
        "denoising_depth": 2,
        "timestep_dim": 32,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        # Shared
        "embedding_dim": 1,
    }

    # ---- Hyperparameter ranges for SA ----
    hyperparam_ranges = {
        "hidden_dim": {
            "range": (16, 128),
            "is_int": True,
            "num_samples": 3,
            "top_k": 2,
        },
        "learning_rate": {
            "range": (1e-5, 3e-3),
            "is_int": False,
            "num_samples": 3,
            "top_k": 2,
        },
        "embedding_dim": {
            "range": (1, 32),
            "is_int": True,
            "num_samples": 3,
            "top_k": 2,
        },
    }

    if transition_type == "mlp":
        hyperparam_ranges.update(
            {
                "width": {
                    "range": (32, 256),
                    "is_int": True,
                    "num_samples": 3,
                    "top_k": 2,
                },
                "depth": {
                    "range": (1, 6),
                    "is_int": True,
                    "num_samples": 3,
                    "top_k": 2,
                },
            }
        )
    else:
        hyperparam_ranges.update(
            {
                "timestep_dim": {
                    "range": (16, 128),
                    "is_int": True,
                    "num_samples": 3,
                    "top_k": 2,
                },
                "num_diffusion_steps": {
                    "range": (2, 6),
                    "is_int": True,
                    "num_samples": 3,
                    "top_k": 2,
                },
                "denoising_hidden_dim": {
                    "range": (32, 128),
                    "is_int": True,
                    "num_samples": 3,
                    "top_k": 2,
                },
                "denoising_depth": {
                    "range": (1, 6),
                    "is_int": True,
                    "num_samples": 3,
                    "top_k": 2,
                },
                "denoising_depth": {
                    "range": (1, 6),
                    "is_int": True,
                    "num_samples": 3,
                    "top_k": 2,
                },
                "beta_start": {
                    "range": (1e-5, 1e-3),
                    "is_int": False,
                    "num_samples": 3,
                    "top_k": 2,
                },
                "beta_end": {
                    "range": (1e-3, 0.1),
                    "is_int": False,
                    "num_samples": 3,
                    "top_k": 2,
                },
            }
        )

    def build_model_fn(config):
        return build_model_from_config(config)

    # -------------------------
    # Sensitivity analysis
    # -------------------------
    print(f"\n=== {transition_type.upper()} :: Sensitivity Analysis ===")
    reduced_ranges = sensitivity_analysis(
        base_config=base_config,
        hyperparam_ranges=hyperparam_ranges,
        build_model_fn=build_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        task_type="regression",
        num_epochs=sa_epochs,
    )

    with open(os.path.join(results_root, "reduced_ranges.json"), "w") as f:
        json.dump(reduced_ranges, f, indent=2)

    # -------------------------
    # Randomized search
    # -------------------------
    print(f"\n=== {transition_type.upper()} :: Randomized Search ===")
    rs_checkpoint_dir = os.path.join(results_root, "checkpoints_rs")
    rs_out = randomized_search(
        base_config=base_config,
        reduced_ranges=reduced_ranges,
        build_model_fn=build_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        task_type="regression",
        num_epochs=rs_epochs,
        num_trials=rs_trials,
        checkpoint_dir=rs_checkpoint_dir,
        experiment_name=f"sine_{transition_type}",
    )

    best_config = rs_out["best_config"]
    best_val_metric_rs = rs_out["best_val_metric"]

    with open(os.path.join(results_root, "best_config_rs.json"), "w") as f:
        json.dump(
            {"best_config": best_config, "best_val_metric_rs": best_val_metric_rs},
            f,
            indent=2,
        )

    # -------------------------
    # Final training from scratch (50 epochs)
    # -------------------------
    print(f"\n=== {transition_type.upper()} :: Final Training (50 epochs) ===")
    final_ckpt_dir = os.path.join(results_root, "checkpoints_final")
    history, best_epoch, best_val_metric_final = train_final_model(
        config=best_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=final_epochs,
        checkpoints_dir=final_ckpt_dir,
    )

    # Load best_model.pt for evaluation
    best_model_path = os.path.join(final_ckpt_dir, "best_model.pt")
    final_model = build_model_from_config(best_config).to(device)
    final_model.load_state_dict(torch.load(best_model_path, map_location=device))

    # -------------------------
    # Metrics on test set
    # -------------------------
    test_mse, test_mae = compute_test_metrics(final_model, test_loader, device)
    horizons = [1, 5, 10, 20, 30, 40, 50]
    horizon_mse = compute_horizon_errors(final_model, test_loader, device, horizons)
    spectral_mse = compute_spectral_error(final_model, test_loader, device)

    metrics = {
        "best_epoch_final": best_epoch,
        "best_val_metric_final": float(best_val_metric_final),
        "best_val_metric_rs": float(best_val_metric_rs),
        "test_mse": test_mse,
        "test_mae": test_mae,
        "horizon_mse": horizon_mse,
        "spectral_mse": spectral_mse,
    }

    with open(os.path.join(results_root, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(results_root, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # -------------------------
    # Plots
    # -------------------------
    plots_dir = os.path.join(results_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_loss_curves(
        history,
        title=f"Sinewave {transition_type.upper()} Loss Curves",
        save_path=os.path.join(plots_dir, "loss_curves.png"),
    )

    plot_example_forecast(
        final_model,
        test_loader,
        device,
        title=f"Sinewave {transition_type.upper()} Example Forecast",
        save_path=os.path.join(plots_dir, "example_forecast.png"),
    )

    plot_horizon_errors(
        horizon_mse,
        title=f"Sinewave {transition_type.upper()} Horizon MSE",
        save_path=os.path.join(plots_dir, "horizon_mse.png"),
    )

    return {
        "transition_type": transition_type,
        "best_config": best_config,
        "metrics": metrics,
        "history": history,
    }


# -----------------------------
# Top-level comparison
# -----------------------------

def main():
    set_seed(0)

    train_loader, val_loader, test_loader = make_sine_dataloaders(
        num_samples=3000,
        seq_len=50,
        batch_size=64,
    )

    results_root = "./results/sinewave"
    os.makedirs(results_root, exist_ok=True)

    diff_results = run_experiment_for_transition(
        transition_type="diffusion",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        results_root=os.path.join(results_root, "diffusion"),
    )

    mlp_results = run_experiment_for_transition(
        transition_type="mlp",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        results_root=os.path.join(results_root, "mlp"),
    )

    # Combined summary + comparison plots
    comparison_dir = os.path.join(results_root, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    summary = {
        "mlp": mlp_results["metrics"],
        "diffusion": diff_results["metrics"],
    }

    with open(os.path.join(comparison_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Test MSE comparison
    labels = ["MLP", "Diffusion"]
    test_mses = [summary["mlp"]["test_mse"], summary["diffusion"]["test_mse"]]

    plt.figure()
    plt.bar(labels, test_mses)
    plt.ylabel("Test MSE (lower is better)")
    plt.title("Sinewave Forecasting: MLP vs Diffusion (Final Models)")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "test_mse_comparison.png"))
    plt.close()

    # Horizon MSE comparison
    horizons = sorted(mlp_results["metrics"]["horizon_mse"].keys())
    mlp_h = [mlp_results["metrics"]["horizon_mse"][h] for h in horizons]
    diff_h = [diff_results["metrics"]["horizon_mse"][h] for h in horizons]

    plt.figure()
    plt.plot(horizons, mlp_h, marker="o", label="MLP")
    plt.plot(horizons, diff_h, marker="o", label="Diffusion")
    plt.xlabel("Horizon (steps)")
    plt.ylabel("MSE")
    plt.title("Horizon-wise MSE: MLP vs Diffusion")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "horizon_mse_comparison.png"))
    plt.close()

    # Spectral error comparison
    plt.figure()
    plt.bar(labels, [summary["mlp"]["spectral_mse"], summary["diffusion"]["spectral_mse"]])
    plt.ylabel("Spectral MSE (lower is better)")
    plt.title("Spectral Error: MLP vs Diffusion")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "spectral_mse_comparison.png"))
    plt.close()

    print("\n=== Finished Sinewave Comparison ===")
    print("MLP metrics:", summary["mlp"])
    print("Diffusion metrics:", summary["diffusion"])
    print(f"Results saved under: {results_root}")


if __name__ == "__main__":
    main()

