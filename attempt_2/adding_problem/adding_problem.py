import os
import json
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
)

from training_utils import (
    build_model_from_config,
    sensitivity_analysis,
    randomized_search,
    train_final_model,
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
# Adding Problem Dataset
# -----------------------------

class AddingProblemDataset(Dataset):
    """
    Classic adding problem:
      Input:  [seq_len, 2]
        channel 0: random uniform noise
        channel 1: two positions marked with 1.0
      Target: scalar sum of the two marked positions
    """

    def __init__(self, num_samples: int, seq_len: int):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len

        self.inputs, self.targets = self._generate()

    def _generate(self):
        xs = []
        ys = []

        for _ in range(self.num_samples):
            x = torch.zeros(self.seq_len, 2)
            x[:, 0] = torch.rand(self.seq_len)

            idx1, idx2 = random.sample(range(self.seq_len), 2)
            x[idx1, 1] = 1.0
            x[idx2, 1] = 1.0

            y = x[idx1, 0] + x[idx2, 0]  # scalar regression target

            xs.append(x)
            ys.append(torch.tensor([y], dtype=torch.float32))

        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_adding_dataloaders(
    num_samples=5000,
    seq_len=50,
    batch_size=64,
    train_frac=0.7,
    val_frac=0.15,
):
    dataset = AddingProblemDataset(num_samples=num_samples, seq_len=seq_len)

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
# Loss + metric for regression → classification
# -----------------------------

def loss_fn(pred, target):
    pred = pred[:, -1, :]

    if target.dim() == 1:
        target = target.unsqueeze(-1)

    return nn.functional.mse_loss(pred, target)

def metric_fn(pred, target):
    pred = pred[:, -1, :]

    if target.dim() == 1:
        target = target.unsqueeze(-1)

    return -nn.functional.mse_loss(pred, target).item()


# -----------------------------
# Evaluation Metrics
# -----------------------------

# -----------------------------
# Correct Regression Metrics
# -----------------------------

def compute_test_metrics(model, test_loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)[:, -1, :]  # final timestep prediction
            preds.append(out.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds).squeeze(-1).numpy()
    targets = torch.cat(targets).squeeze(-1).numpy()

    # --- Core regression metrics ---
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))
    r2 = float(1 - np.sum((preds - targets)**2) / np.sum((targets - targets.mean())**2))

    # --- Spectral MSE (FFT-based frequency error) ---
    fft_pred = np.fft.rfft(preds)
    fft_tgt = np.fft.rfft(targets)
    spectral_mse = float(np.mean(np.abs(fft_pred - fft_tgt) ** 2))

    # --- Error distribution stats ---
    median_ae = float(np.median(np.abs(preds - targets)))
    max_ae = float(np.max(np.abs(preds - targets)))

    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spectral_mse": spectral_mse,
        "median_abs_error": median_ae,
        "max_abs_error": max_ae,
    }

    return metrics, preds, targets

# -----------------------------
# Plotting
# -----------------------------

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


def plot_example_prediction(model, test_loader, device, title, save_path):
    model.eval()
    x_ex, y_ex = next(iter(test_loader))
    x_ex = x_ex.to(device)
    y_ex = y_ex.to(device)

    with torch.no_grad():
        pred = model(x_ex)[:, -1, :]

    plt.figure()
    plt.scatter(range(len(pred)), pred.cpu(), label="Prediction")
    plt.scatter(range(len(y_ex)), y_ex.cpu(), label="Target")
    plt.xlabel("Sample")
    plt.ylabel("Sum")
    plt.title(title)
    plt.legend()
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
        "input_dim": 2,
        "output_dim": 1,
        "hidden_dim": 256,
        "transition_type": transition_type,
        "learning_rate": 1e-4,
        "width": 64,
        "depth": 3,
        "out_width": 64,
        "out_depth": 2,
        "num_diffusion_steps": 10,
        "denoising_hidden_dim": 64,
        "denoising_depth": 3,
        "timestep_dim": 32,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "embedding_dim": 16,
    }

    # ---- Hyperparameter ranges ----
    hyperparam_ranges = {
        "hidden_dim": {"range": (16, 512), "is_int": True, "num_samples": 3, "top_k": 2},
        "learning_rate": {"range": (1e-6, 3e-4), "is_int": False, "num_samples": 3, "top_k": 2},
        "embedding_dim": {"range": (1, 64), "is_int": True, "num_samples": 3, "top_k": 2},
    }

    if transition_type == "mlp":
        hyperparam_ranges.update({
            "width": {"range": (32, 256), "is_int": True, "num_samples": 3, "top_k": 2},
            "depth": {"range": (1, 6), "is_int": True, "num_samples": 3, "top_k": 2},
        })
    else:
        hyperparam_ranges.update({
            "timestep_dim": {"range": (16, 128), "is_int": True, "num_samples": 3, "top_k": 2},
            "num_diffusion_steps": {"range": (2, 32), "is_int": True, "num_samples": 3, "top_k": 2},
            "denoising_hidden_dim": {"range": (32, 256), "is_int": True, "num_samples": 3, "top_k": 2},
            "denoising_depth": {"range": (1, 6), "is_int": True, "num_samples": 3, "top_k": 2},
            "beta_start": {"range": (1e-5, 1e-3), "is_int": False, "num_samples": 3, "top_k": 2},
            "beta_end": {"range": (1e-3, 0.1), "is_int": False, "num_samples": 3, "top_k": 2},
        })

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
        num_epochs=rs_epochs,
        num_trials=rs_trials,
        checkpoint_dir=rs_checkpoint_dir,
        experiment_name=f"adding_{transition_type}",
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
    # Final training
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
        loss_fn=loss_fn,
        metric_fn=metric_fn
    )

    # Load best model
    best_model_path = os.path.join(final_ckpt_dir, "best_model.pt")
    final_model = build_model_from_config(best_config).to(device)
    final_model.load_state_dict(torch.load(best_model_path, map_location=device))

    # -------------------------
    # Test metrics
    # -------------------------
    metrics, preds, targets = compute_test_metrics(final_model, test_loader, device)

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
        title=f"Adding {transition_type.upper()} Loss Curves",
        save_path=os.path.join(plots_dir, "loss_curves.png"),
    )

    plot_example_prediction(
        final_model,
        test_loader,
        device,
        title=f"Adding {transition_type.upper()} Example Prediction",
        save_path=os.path.join(plots_dir, "example_prediction.png"),
    )

    # Residual plot
    plt.figure()
    residuals = preds - targets
    plt.scatter(targets, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Target")
    plt.ylabel("Residual (Pred - Target)")
    plt.title(f"Adding {transition_type.upper()} Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "residual_plot.png"))
    plt.close()


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

    train_loader, val_loader, test_loader = make_adding_dataloaders(
        num_samples=10000,
        seq_len=32,
        batch_size=64,
    )

    results_root = "./results/adding"
    os.makedirs(results_root, exist_ok=True)

#    mlp_results = run_experiment_for_transition(
#        transition_type="mlp",
#        train_loader=train_loader,
#        val_loader=val_loader,
#        test_loader=test_loader,
#        results_root=os.path.join(results_root, "mlp"),
#    )

    diff_results = run_experiment_for_transition(
        transition_type="diffusion",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        results_root=os.path.join(results_root, "diffusion"),
    )

    # Combined summary
    comparison_dir = os.path.join(results_root, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    summary = {
        "mlp": mlp_results["metrics"],
        "diffusion": diff_results["metrics"],
    }

    with open(os.path.join(comparison_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Accuracy comparison
    labels = ["MLP", "Diffusion"]
    mses = [summary["mlp"]["mse"], summary["diffusion"]["mse"]]

    plt.figure()
    plt.bar(labels, mses)
    plt.ylabel("MSE")
    plt.title("Adding Problem: MLP vs Diffusion (Regression)")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "mse_comparison.png"))
    plt.close()

    print("\n=== Finished Adding Problem Comparison ===")
    print("MLP metrics:", summary["mlp"])
    print("Diffusion metrics:", summary["diffusion"])
    print(f"Results saved under: {results_root}")


if __name__ == "__main__":
    main()

