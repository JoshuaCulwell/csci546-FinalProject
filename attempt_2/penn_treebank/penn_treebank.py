import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
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


# ============================================================
#                PENN TREEBANK CHARACTER DATASET
# ============================================================

class PTBDataset(Dataset):
    """
    Character-level Penn Treebank.
    Produces sequences of length seq_len with next-char prediction.
    """

    def __init__(self, text_path, seq_len=128, max_chars=None, stoi=None, itos=None):
        with open(text_path, "r") as f:
            raw = f.read()

        if max_chars is not None:
            raw = raw[:max_chars]

        # Build vocabulary
        if stoi is None:
            chars = sorted(list(set(raw)))
            chars.append("<unk>")
            self.stoi = {c: i for i, c in enumerate(chars)}
            self.itos = {i: c for c, i in self.stoi.items()}
        else:
            self.stoi = stoi
            self.itos = itos

        self.vocab_size = len(self.stoi)

        # Encode entire text
        encoded = torch.tensor(
            [self.stoi[c] if c in self.stoi else self.stoi["<unk>"] for c in raw],
            dtype=torch.long
        )

        self.seq_len = seq_len
        self.data = encoded

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]#.unsqueeze(-1) #.float()
        y = self.data[idx + 1 : idx + self.seq_len + 1]  # next char
        return x, y


def make_ptb_dataloaders(
    seq_len=128,
    batch_size=2048,
    root="./datasets",
):
    train_ds = PTBDataset(os.path.join(root, "ptb.train.txt"), seq_len, 50000)
    stoi, itos = train_ds.stoi, train_ds.itos

    val_ds   = PTBDataset(os.path.join(root, "ptb.valid.txt"), seq_len, 10000, stoi=stoi, itos=itos)
    test_ds  = PTBDataset(os.path.join(root, "ptb.test.txt"), seq_len, 10000, stoi=stoi, itos=itos)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds.vocab_size


# ============================================================
#                LOSS + METRIC (CLASSIFICATION)
# ============================================================

def loss_fn(pred, target):
    """
    pred: [B, T, vocab]
    target: [B, T]
    """
    pred = pred.reshape(-1, pred.size(-1))
    target = target.reshape(-1)
    return nn.functional.cross_entropy(pred, target)


def metric_fn(pred, target):
    """
    Returns accuracy for validation metric.
    """
    pred = pred.reshape(-1, pred.size(-1))
    target = target.reshape(-1)
    preds = pred.argmax(dim=-1)
    acc = (preds == target).float().mean().item()
    return acc


# ============================================================
#                TEST METRICS (ACCURACY, PREC, REC, F1, BRIER)
# ============================================================

def compute_test_metrics(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)  # [B, T, vocab]
            probs = torch.softmax(out, dim=-1)

            preds = probs.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_probs.append(probs.cpu())

    preds = torch.cat(all_preds).reshape(-1).numpy()
    targets = torch.cat(all_targets).reshape(-1).numpy()
    probs = torch.cat(all_probs).reshape(-1, probs.size(-1)).numpy()

    # Brier score requires probability of true class
    true_class_probs = probs[np.arange(len(targets)), targets]

    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(targets, preds, average="macro", zero_division=0)),
        "f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "brier": float(brier_score_loss(
            targets,
            probs,
            labels=list(range(probs.shape[1]))
        )),
    }

    return metrics, preds, targets


# ============================================================
#                PLOTTING (UNCHANGED)
# ============================================================

def plot_loss_curves(history, title, save_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_example_prediction(model, test_loader, device, title, save_path):
    """
    For PTB, we visualize predicted vs true characters for a single sequence.
    """
    model.eval()
    x_ex, y_ex = next(iter(test_loader))
    x_ex = x_ex.to(device)
    y_ex = y_ex.to(device)

    with torch.no_grad():
        out = model(x_ex)
        preds = out.argmax(dim=-1)

    plt.figure(figsize=(10, 4))
    plt.plot(preds[0].cpu(), label="Predicted IDs")
    plt.plot(y_ex[0].cpu(), label="Target IDs")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================
#                EXPERIMENT LOOP (UNCHANGED)
# ============================================================

def run_experiment_for_transition(
    transition_type: str,
    train_loader,
    val_loader,
    test_loader,
    vocab_size,
    results_root: str,
    sa_epochs: int = 3,
    rs_epochs: int = 10,
    rs_trials: int = 8,
    final_epochs: int = 50,
    final_metrics_only = False
):
    os.makedirs(results_root, exist_ok=True)

    # ---- Base config ----
    base_config = {
        "input_dim": 1,
        "output_dim": vocab_size,
        "hidden_dim": 512,
        "transition_type": transition_type,
        "learning_rate": 1e-4,
        "width": 256,
        "depth": 3,
        "out_width": 256,
        "out_depth": 2,
        "num_diffusion_steps": 3,
        "denoising_hidden_dim": 256,
        "denoising_depth": 3,
        "timestep_dim": 32,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "embedding_dim": 128,
        "vocab_size": vocab_size,
    }

    # ---- Hyperparameter ranges ----
    hyperparam_ranges = {
        "hidden_dim": {"range": (128, 1024), "is_int": True, "num_samples": 3, "top_k": 2},
        "learning_rate": {"range": (1e-6, 3e-4), "is_int": False, "num_samples": 3, "top_k": 2},
        "embedding_dim": {"range": (16, 1024), "is_int": True, "num_samples": 3, "top_k": 2},
    }

    if transition_type == "mlp":
        hyperparam_ranges.update({
            "width": {"range": (32, 512), "is_int": True, "num_samples": 3, "top_k": 2},
            "depth": {"range": (1, 6), "is_int": True, "num_samples": 3, "top_k": 2},
        })
    else:
        hyperparam_ranges.update({
            "timestep_dim": {"range": (16, 128), "is_int": True, "num_samples": 3, "top_k": 2},
            "num_diffusion_steps": {"range": (2, 5), "is_int": True, "num_samples": 3, "top_k": 2},
            "denoising_hidden_dim": {"range": (32, 512), "is_int": True, "num_samples": 3, "top_k": 2},
            "denoising_depth": {"range": (1, 6), "is_int": True, "num_samples": 3, "top_k": 2},
            #"beta_start": {"range": (1e-5, 1e-3), "is_int": False, "num_samples": 3, "top_k": 2},
            #"beta_end": {"range": (1e-3, 0.1), "is_int": False, "num_samples": 3, "top_k": 2},
        })

    def build_model_fn(config):
        return build_model_from_config(config)

    # -------------------------
    # Sensitivity analysis
    # -------------------------
    print(f"\n=== {transition_type.upper()} :: Sensitivity Analysis ===")
    if final_metrics_only:
        print("skipping...")
    else:
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
    
    if final_metrics_only:
        print("skipping...")
    else:
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
            experiment_name=f"ptb_{transition_type}",
        )

        best_config = rs_out["best_config"]
        best_val_metric_rs = rs_out["best_val_metric"]

        with open(os.path.join(results_root, "best_config_rs.json"), "w") as f:
            json.dump(
                {"best_config": best_config, "best_val_metric_rs": best_val_metric_rs},
                f,
                indent=2,
            )

    with open(os.path.join(results_root, "best_config_rs.json"), "r") as f:
        best_config = json.load(f)["best_config"]

    # -------------------------
    # Final training
    # -------------------------
    print(f"\n=== {transition_type.upper()} :: Final Training ===")
    final_ckpt_dir = os.path.join(results_root, "checkpoints_final")

    if final_metrics_only:
        print("skipping...")
    else:
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

    if final_metrics_only:
        print("skipping...")
    else:
        with open(os.path.join(results_root, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # -------------------------
    # Plots
    # -------------------------
    plots_dir = os.path.join(results_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if final_metrics_only:
        print("skipping...")
    else:
        plot_loss_curves(
            history,
            title=f"PTB {transition_type.upper()} Loss Curves",
            save_path=os.path.join(plots_dir, "loss_curves.png"),
        )

    plot_example_prediction(
        final_model,
        test_loader,
        device,
        title=f"PTB {transition_type.upper()} Example Prediction",
        save_path=os.path.join(plots_dir, "example_prediction.png"),
    )

    return {
        "transition_type": transition_type,
        "best_config": best_config,
        "metrics": metrics,
        "history": history,
    }


# ============================================================
#                TOP-LEVEL COMPARISON
# ============================================================

def main():
    set_seed(0)

    train_loader, val_loader, test_loader, vocab_size = make_ptb_dataloaders(
        seq_len=128,
        batch_size=256,
        root="./datasets",
    )

    results_root = "./results/penn_treebank"
    os.makedirs(results_root, exist_ok=True)

#    mlp_results = run_experiment_for_transition(
#        transition_type="mlp",
#        train_loader=train_loader,
#        val_loader=val_loader,
#        test_loader=test_loader,
#        vocab_size=vocab_size,
#        results_root=os.path.join(results_root, "mlp"),
#    )
    diff_results = run_experiment_for_transition(
        transition_type="diffusion",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        vocab_size=vocab_size,
        results_root=os.path.join(results_root, "diffusion"),
    )

    # Combined summary
    comparison_dir = os.path.join(results_root, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    #summary = {
        # "mlp": mlp_results["metrics"],
        #"diffusion": diff_results["metrics"],
    #}

    #with open(os.path.join(comparison_dir, "summary.json"), "w") as f:
    #    json.dump(summary, f, indent=2)

    #print("\n=== Finished PTB Comparison ===")
    #print("Diffusion metrics:", summary["diffusion"])
    print(f"Results saved under: {results_root}")


if __name__ == "__main__":
    main()

