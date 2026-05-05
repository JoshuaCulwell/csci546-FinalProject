import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from generic_experiment import (
    sensitivity_analysis,
    randomized_search,
    build_model,
)

# -----------------------------
# Repro + device
# -----------------------------

def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Adding Problem Dataset
# -----------------------------

class AddingDataset(Dataset):
    """
    Each sample:
        input:  [seq_len, 2]
            channel 0 = random uniform noise
            channel 1 = two positions marked with 1.0
        target: scalar sum of the two marked noise values
    """

    def __init__(self, num_samples, seq_len=50):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len

        self.inputs, self.targets = self._generate()

    def _generate(self):
        xs = []
        ys = []

        for _ in range(self.num_samples):
            seq = torch.rand(self.seq_len)
            markers = torch.zeros(self.seq_len)
            idx1, idx2 = random.sample(range(self.seq_len), 2)
            markers[idx1] = 1.0
            markers[idx2] = 1.0

            x = torch.stack([seq, markers], dim=-1)  # [T, 2]
            y = seq[idx1] + seq[idx2]                # scalar

            xs.append(x)
            ys.append(torch.tensor([y], dtype=torch.float32))

        return torch.stack(xs), torch.stack(ys)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_adding_dataloaders(
    num_samples=50000,
    seq_len=50,
    batch_size=128,
    train_frac=0.7,
    val_frac=0.15,
):
    dataset = AddingDataset(num_samples=num_samples, seq_len=seq_len)

    n_train = int(train_frac * num_samples)
    n_val = int(val_frac * num_samples)
    n_test = num_samples - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# -----------------------------
# Loss + metric
# -----------------------------

loss_fn = nn.MSELoss()

def metric_fn(pred, target):
    return -nn.functional.mse_loss(pred, target).item()


# -----------------------------
# Build model from config
# -----------------------------

def build_model_from_config(config):
    transition_type = config["transition_type"]

    if transition_type == "mlp":
        transition_hparams = {
            "width": config["width"],
            "depth": config["depth"],
        }
    else:
        transition_hparams = {
            "num_diffusion_steps": config["num_diffusion_steps"],
            "denoising_hidden_dim": config["denoising_hidden_dim"],
            "denoising_depth": config["denoising_depth"],
            "denoising_activation_function": nn.ReLU,
            "timestep_dim": config["timestep_dim"],
            "beta_start": config["beta_start"],
            "beta_end": config["beta_end"],
        }

    out_hparams = {
        "width": config["out_width"],
        "depth": config["out_depth"],
    }

    return build_model(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        hidden_dim=config["hidden_dim"],
        transition_type=transition_type,
        transition_hparams=transition_hparams,
        out_mlp_hparams=out_hparams,
    )


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

    history = {"train_loss": [], "val_loss": [], "val_metric": []}
    best_val_metric = -float("inf")
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
            logits = model(x)[:, -1, :]  # last timestep output
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x.size(0)
            total_train_count += x.size(0)

        avg_train_loss = total_train_loss / total_train_count

        # ---- Val ----
        model.eval()
        total_val_loss = 0.0
        total_val_metric = 0.0
        total_val_count = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)[:, -1, :]
                loss = loss_fn(logits, y)
                metric = metric_fn(logits, y)

                total_val_loss += loss.item() * x.size(0)
                total_val_metric += metric * x.size(0)
                total_val_count += x.size(0)

        avg_val_loss = total_val_loss / total_val_count
        avg_val_metric = total_val_metric / total_val_count

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_metric"].append(avg_val_metric)

        # Save epoch checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(checkpoints_dir, f"epoch_{epoch+1:03d}.pt"),
        )

        # Save best model
        if avg_val_metric > best_val_metric:
            best_val_metric = avg_val_metric
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_dir, "best_model.pt"),
            )

        print(
            f"[Final Train] Epoch {epoch+1}/{num_epochs} "
            f"TrainLoss={avg_train_loss:.4f} "
            f"ValLoss={avg_val_loss:.4f} "
            f"ValMetric={avg_val_metric:.4f}"
        )

    return history, best_epoch, best_val_metric


# -----------------------------
# Test metrics
# -----------------------------

def compute_test_metrics(model, test_loader, device):
    model.eval()
    mse_list = []
    mae_list = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)[:, -1, :]
            mse_list.append(nn.functional.mse_loss(pred, y).item())
            mae_list.append(nn.functional.l1_loss(pred, y).item())

    return float(sum(mse_list)/len(mse_list)), float(sum(mae_list)/len(mae_list))


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


# -----------------------------
# Full experiment per transition
# -----------------------------

def run_experiment_for_transition(
    transition_type,
    train_loader,
    val_loader,
    test_loader,
    results_root,
    sa_epochs=2,
    rs_epochs=5,
    rs_trials=6,
    final_epochs=50,
):
    os.makedirs(results_root, exist_ok=True)

    # ---- Base config ----
    base_config = {
        "input_dim": 2,
        "output_dim": 1,
        "hidden_dim": 32,
        "transition_type": transition_type,
        "learning_rate": 1e-3,
        "width": 64,
        "depth": 2,
        "out_width": 64,
        "out_depth": 2,
        "num_diffusion_steps": 4,
        "denoising_hidden_dim": 64,
        "denoising_depth": 2,
        "timestep_dim": 32,
        "beta_start": 1e-4,
        "beta_end": 0.02,
    }

    # ---- Hyperparameter ranges ----
    hyperparam_ranges = {
        "hidden_dim": {"range": (16, 64), "is_int": True, "num_samples": 3, "top_k": 2},
        "learning_rate": {"range": (1e-4, 5e-3), "is_int": False, "num_samples": 3, "top_k": 2},
    }

    if transition_type == "mlp":
        hyperparam_ranges.update({
            "width": {"range": (32, 128), "is_int": True, "num_samples": 3, "top_k": 2},
            "depth": {"range": (1, 3), "is_int": True, "num_samples": 3, "top_k": 2},
        })
    else:
        hyperparam_ranges.update({
            "num_diffusion_steps": {"range": (2, 6), "is_int": True, "num_samples": 3, "top_k": 2},
            "denoising_hidden_dim": {"range": (32, 128), "is_int": True, "num_samples": 3, "top_k": 2},
        })

    def build_model_fn(config):
        return build_model_from_config(config)

    # -------------------------
    # Sensitivity Analysis
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
    # Randomized Search
    # -------------------------
    print(f"\n=== {transition_type.upper()} :: Randomized Search ===")
    rs_ckpt_dir = os.path.join(results_root, "checkpoints_rs")
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
        checkpoint_dir=rs_ckpt_dir,
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
    # Final Training (50 epochs)
    # -------------------------
    print(f"\n=== {transition_type.upper()} :: Final Training ===")
    final_ckpt_dir = os.path.join(results_root, "checkpoints_final")
    history, best_epoch, best_val_metric_final = train_final_model(
        config=best_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=final_epochs,
        checkpoints_dir=final_ckpt_dir,
    )

    # Load best model
    best_model_path = os.path.join(final_ckpt_dir, "best_model.pt")
    final_model = build_model_from_config(best_config).to(device)
    final_model.load_state_dict(torch.load(best_model_path, map_location=device))

    # -------------------------
    # Test metrics
    # -------------------------
    test_mse, test_mae = compute_test_metrics(final_model, test_loader, device)

    metrics = {
        "best_epoch_final": best_epoch,
        "best_val_metric_final": float(best_val_metric_final),
        "best_val_metric_rs": float(best_val_metric_rs),
        "test_mse": test_mse,
        "test_mae": test_mae,
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
        title=f"Adding {transition_type.upper()} Loss Curves",
        save_path=os.path.join(plots_dir, "loss_curves.png"),
    )

    return {
        "transition_type": transition_type,
        "best_config": best_config,
        "metrics": metrics,
        "history": history,
    }


# -----------------------------
# Main
# -----------------------------

def main():
    set_seed(0)

    train_loader, val_loader, test_loader = make_adding_dataloaders(
        num_samples=50000,
        seq_len=50,
        batch_size=128,
    )

    results_root = "./results/adding"
    os.makedirs(results_root, exist_ok=True)

    mlp_results = run_experiment_for_transition(
        "mlp",
        train_loader,
        val_loader,
        test_loader,
        results_root=os.path.join(results_root, "mlp"),
    )

    diff_results = run_experiment_for_transition(
        "diffusion",
        train_loader,
        val_loader,
        test_loader,
        results_root=os.path.join(results_root, "diffusion"),
    )

    # Comparison summary
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
    plt.ylabel("Test MSE")
    plt.title("Adding Problem: MLP vs Diffusion")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "test_mse_comparison.png"))
    plt.close()

    print("\n=== Finished Adding Problem Comparison ===")
    print("MLP:", summary["mlp"])
    print("Diffusion:", summary["diffusion"])
    print(f"Results saved under: {results_root}")


if __name__ == "__main__":
    main()

