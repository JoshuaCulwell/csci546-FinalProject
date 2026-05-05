# generic_experiment.py  (tqdm‑enabled version)

import os
import json
import math
import random
from copy import deepcopy
from typing import Dict, Any, Callable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import VariableTransitionRNN, MLPTransition, DiffusionTransition


# -----------------------------
# Utility: checkpoint + logging
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_config(config: Dict[str, Any], save_dir: str, name: str):
    ensure_dir(save_dir)
    path = os.path.join(save_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


# -----------------------------
# Model construction
# -----------------------------

def build_transition_module(
    transition_type: str,
    hidden_dim: int,
    input_dim: int,
    transition_hparams: Dict[str, Any],
) -> nn.Module:

    if transition_type == "mlp":
        return MLPTransition(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            input_embedding_dim=transition_hparams["embedding_dim"],
            width=transition_hparams["width"],
            depth=transition_hparams["depth"],
            activation_function=transition_hparams.get("activation_function", nn.ReLU),
        )

    elif transition_type == "diffusion":
        return DiffusionTransition(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            input_embedding_dim=transition_hparams["embedding_dim"],
            num_diffusion_steps=transition_hparams["num_diffusion_steps"],
            denoising_hidden_dim=transition_hparams["denoising_hidden_dim"],
            denoising_depth=transition_hparams["denoising_depth"],
            denoising_activation_function=transition_hparams.get(
                "denoising_activation_function", nn.ReLU
            ),
            timestep_dim=transition_hparams["timestep_dim"],
            beta_start=transition_hparams["beta_start"],
            beta_end=transition_hparams["beta_end"],
        )

    else:
        raise ValueError(f"Unknown transition_type: {transition_type}")


def build_model(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    transition_type: str,
    transition_hparams: Dict[str, Any],
    out_mlp_hparams: Dict[str, Any],
) -> nn.Module:

    transition_module = build_transition_module(
        transition_type=transition_type,
        hidden_dim=hidden_dim,
        input_dim=input_dim,
        transition_hparams=transition_hparams,
    )

    model = VariableTransitionRNN(
        in_dim=input_dim,
        out_dim=output_dim,
        hidden_state_dim=hidden_dim,
        transition_module=transition_module,
        out_mlp_width=out_mlp_hparams["width"],
        out_mlp_depth=out_mlp_hparams["depth"],
        out_mlp_activation_function=out_mlp_hparams.get("activation_function", nn.ReLU),
    )
    return model


# -----------------------------
# Training / evaluation
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    task_type: str,
) -> float:

    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)

        if task_type == "classification":
            loss = loss_fn(logits[:, -1, :], y[:, -1])
        else:
            loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    metric_fn: Callable,
    device: torch.device,
    task_type: str,
) -> Tuple[float, float]:

    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    total_count = 0

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        if task_type == "classification":
            loss = loss_fn(logits[:, -1, :], y[:, -1])
            preds = logits[:, -1, :].argmax(dim=-1)
            metric = metric_fn(preds, y[:, -1])
        else:
            loss = loss_fn(logits, y)
            metric = metric_fn(logits, y)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_metric += metric * batch_size
        total_count += batch_size

    return (
        total_loss / max(total_count, 1),
        total_metric / max(total_count, 1),
    )


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    metric_fn: Callable,
    device: torch.device,
    task_type: str,
    num_epochs: int,
) -> Dict[str, Any]:

    model.to(device)
    best_val_metric = -math.inf
    best_state = None

    history = {"train_loss": [], "val_loss": [], "val_metric": []}

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, task_type
        )
        val_loss, val_metric = evaluate(
            model, val_loader, loss_fn, metric_fn, device, task_type
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_state = deepcopy(model.state_dict())

    return {
        "best_val_metric": best_val_metric,
        "best_state_dict": best_state,
        "history": history,
    }


# -----------------------------
# Sensitivity analysis
# -----------------------------

def sample_from_range(rng: Tuple[float, float], num_samples: int, is_int: bool):
    low, high = rng
    if is_int:
        low, high = int(low), int(high)
        return sorted(random.sample(range(low, high + 1), k=num_samples))
    else:
        return [low + (high - low) * i / (num_samples - 1) for i in range(num_samples)]


def sensitivity_analysis(
    base_config: Dict[str, Any],
    hyperparam_ranges: Dict[str, Dict[str, Any]],
    build_model_fn: Callable[[Dict[str, Any]], nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable,
    metric_fn: Callable,
    device: torch.device,
    task_type: str,
    num_epochs: int,
) -> Dict[str, Dict[str, Any]]:

    reduced_ranges = {}

    for param_name, spec in hyperparam_ranges.items():
        print(f"\n[SA] Hyperparameter: {param_name}")

        values = sample_from_range(
            spec["range"], spec["num_samples"], spec.get("is_int", False)
        )

        results = []

        for v in tqdm(values, desc=f"SA {param_name}"):
            config = deepcopy(base_config)
            config[param_name] = v

            model = build_model_fn(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

            out = run_training_loop(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                metric_fn=metric_fn,
                device=device,
                task_type=task_type,
                num_epochs=num_epochs,
            )

            results.append((v, out["best_val_metric"]))
            print(f"  value={v} -> metric={out['best_val_metric']:.4f}")

        results.sort(key=lambda x: x[1], reverse=True)
        top_k = spec.get("top_k", max(2, len(results) // 3))
        top_values = [v for v, _ in results[:top_k]]

        reduced_ranges[param_name] = {
            "range": (min(top_values), max(top_values)),
            "is_int": spec.get("is_int", False),
        }

        print(f"[SA] Reduced range for {param_name}: {reduced_ranges[param_name]['range']}")

    return reduced_ranges


# -----------------------------
# Randomized search
# -----------------------------

def sample_config_from_ranges(
    base_config: Dict[str, Any],
    reduced_ranges: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:

    config = deepcopy(base_config)
    for param_name, spec in reduced_ranges.items():
        low, high = spec["range"]
        if spec.get("is_int", False):
            config[param_name] = random.randint(int(low), int(high))
        else:
            config[param_name] = random.uniform(low, high)
    return config


def randomized_search(
    base_config: Dict[str, Any],
    reduced_ranges: Dict[str, Dict[str, Any]],
    build_model_fn: Callable[[Dict[str, Any]], nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable,
    metric_fn: Callable,
    device: torch.device,
    task_type: str,
    num_epochs: int,
    num_trials: int,
    checkpoint_dir: str,
    experiment_name: str,
) -> Dict[str, Any]:

    ensure_dir(checkpoint_dir)

    best_metric = -math.inf
    best_config = None
    best_state = None

    for trial in tqdm(range(num_trials), desc="Randomized Search"):
        config = sample_config_from_ranges(base_config, reduced_ranges)
        print(f"\n[RS] Trial {trial + 1}/{num_trials} with config: {config}")

        model = build_model_fn(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        out = run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            device=device,
            task_type=task_type,
            num_epochs=num_epochs,
        )

        metric = out["best_val_metric"]
        print(f"[RS] Trial {trial + 1} metric: {metric:.4f}")

        trial_name = f"{experiment_name}_trial_{trial + 1}"
        save_config(
            {"config": config, "best_val_metric": metric, "history": out["history"]},
            checkpoint_dir,
            trial_name,
        )

        if metric > best_metric:
            best_metric = metric
            best_config = config
            best_state = out["best_state_dict"]

    final_name = f"{experiment_name}_best"
    torch.save(best_state, os.path.join(checkpoint_dir, f"{final_name}_model.pt"))
    save_config(
        {"best_config": best_config, "best_val_metric": best_metric},
        checkpoint_dir,
        final_name,
    )

    return {
        "best_config": best_config,
        "best_val_metric": best_metric,
        "best_state_dict": best_state,
    }


# -----------------------------
# FULL TEST BLOCK (unchanged)
# -----------------------------
if __name__ == "__main__":
    """
    This block verifies that:
      • model construction works
      • sensitivity analysis runs
      • randomized search runs
      • configs are saved to ./checkpoints/
      • training loop executes end‑to‑end

    It uses a tiny synthetic dataset so the script is fully runnable.
    """

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 1. Create a tiny synthetic dataset (regression example)
    # ---------------------------------------------------------
    BATCH = 16
    SEQ = 6
    INPUT_DIM = 8
    OUTPUT_DIM = 4

    X = torch.randn(128, SEQ, INPUT_DIM)
    Y = torch.randn(128, SEQ, OUTPUT_DIM)

    train_ds = TensorDataset(X[:100], Y[:100])
    val_ds   = TensorDataset(X[100:], Y[100:])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH)

    # ---------------------------------------------------------
    # 2. Define loss + metric for regression
    # ---------------------------------------------------------
    loss_fn = torch.nn.MSELoss()

    def metric_fn(pred, target):
        return -torch.nn.functional.mse_loss(pred, target).item()

    task_type = "regression"

    # ---------------------------------------------------------
    # 3. Base hyperparameter config
    # ---------------------------------------------------------
    base_config = {
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "hidden_dim": 16,
        "transition_type": "mlp",   # can switch to "diffusion"
        "learning_rate": 1e-3,

        # transition module hyperparams
        "width": 32,
        "depth": 2,

        # output MLP
        "out_width": 32,
        "out_depth": 2,
    }

    # ---------------------------------------------------------
    # 4. Hyperparameter ranges for sensitivity analysis
    # ---------------------------------------------------------
    hyperparam_ranges = {
        "hidden_dim": {
            "range": (8, 32),
            "is_int": True,
            "num_samples": 3,
            "top_k": 2,
        },
        "learning_rate": {
            "range": (1e-4, 5e-3),
            "is_int": False,
            "num_samples": 3,
            "top_k": 2,
        },
    }

    # ---------------------------------------------------------
    # 5. Model builder wrapper
    # ---------------------------------------------------------
    def build_model_fn(config):
        transition_hparams = {
            "width": config["width"],
            "depth": config["depth"],
        }

        out_hparams = {
            "width": config["out_width"],
            "depth": config["out_depth"],
        }

        return build_model(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
            transition_type=config["transition_type"],
            transition_hparams=transition_hparams,
            out_mlp_hparams=out_hparams,
        )

    # ---------------------------------------------------------
    # 6. Run sensitivity analysis
    # ---------------------------------------------------------
    print("\n=== Running Sensitivity Analysis ===")
    reduced_ranges = sensitivity_analysis(
        base_config=base_config,
        hyperparam_ranges=hyperparam_ranges,
        build_model_fn=build_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        task_type=task_type,
        num_epochs=1,   # small for test
    )

    print("\nReduced ranges:", reduced_ranges)

    # ---------------------------------------------------------
    # 7. Run randomized search
    # ---------------------------------------------------------
    print("\n=== Running Randomized Search ===")
    result = randomized_search(
        base_config=base_config,
        reduced_ranges=reduced_ranges,
        build_model_fn=build_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        task_type=task_type,
        num_epochs=1,      # small for test
        num_trials=2,      # small for test
        checkpoint_dir="./checkpoints",
        experiment_name="generic_test",
    )

    print("\n=== Finished ===")
    print("Best config:", result["best_config"])
    print("Best metric:", result["best_val_metric"])


    """
    Classification test block.
    This verifies:
      • model builds correctly
      • training loop runs for classification
      • sensitivity analysis works
      • randomized search works
      • configs save to ./checkpoints/

    Uses a synthetic classification dataset.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 1. Synthetic classification dataset
    # ---------------------------------------------------------
    BATCH = 16
    SEQ = 6
    INPUT_DIM = 8
    NUM_CLASSES = 5

    X = torch.randn(200, SEQ, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (200, SEQ))  # class per timestep

    train_ds = TensorDataset(X[:150], y[:150])
    val_ds   = TensorDataset(X[150:], y[150:])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH)

    # ---------------------------------------------------------
    # 2. Loss + metric for classification
    # ---------------------------------------------------------
    loss_fn = torch.nn.CrossEntropyLoss()

    def metric_fn(pred, target):
        # pred: [B], target: [B]
        return (pred == target).float().mean().item()

    task_type = "classification"

    # ---------------------------------------------------------
    # 3. Base hyperparameter config
    # ---------------------------------------------------------
    base_config = {
        "input_dim": INPUT_DIM,
        "output_dim": NUM_CLASSES,
        "hidden_dim": 16,
        "transition_type": "mlp",   # can switch to "diffusion"
        "learning_rate": 1e-3,

        # transition module hyperparams
        "width": 32,
        "depth": 2,

        # output MLP
        "out_width": 32,
        "out_depth": 2,
    }

    # ---------------------------------------------------------
    # 4. Hyperparameter ranges for sensitivity analysis
    # ---------------------------------------------------------
    hyperparam_ranges = {
        "hidden_dim": {
            "range": (8, 32),
            "is_int": True,
            "num_samples": 3,
            "top_k": 2,
        },
        "learning_rate": {
            "range": (1e-4, 5e-3),
            "is_int": False,
            "num_samples": 3,
            "top_k": 2,
        },
    }

    # ---------------------------------------------------------
    # 5. Model builder wrapper
    # ---------------------------------------------------------
    def build_model_fn(config):
        transition_hparams = {
            "width": config["width"],
            "depth": config["depth"],
        }

        out_hparams = {
            "width": config["out_width"],
            "depth": config["out_depth"],
        }

        return build_model(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_dim=config["hidden_dim"],
            transition_type=config["transition_type"],
            transition_hparams=transition_hparams,
            out_mlp_hparams=out_hparams,
        )

    # ---------------------------------------------------------
    # 6. Sensitivity analysis
    # ---------------------------------------------------------
    print("\n=== Running Classification Sensitivity Analysis ===")
    reduced_ranges = sensitivity_analysis(
        base_config=base_config,
        hyperparam_ranges=hyperparam_ranges,
        build_model_fn=build_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        task_type=task_type,
        num_epochs=1,   # small for test
    )

    print("\nReduced ranges:", reduced_ranges)

    # ---------------------------------------------------------
    # 7. Randomized search
    # ---------------------------------------------------------
    print("\n=== Running Classification Randomized Search ===")
    result = randomized_search(
        base_config=base_config,
        reduced_ranges=reduced_ranges,
        build_model_fn=build_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
        task_type=task_type,
        num_epochs=1,      # small for test
        num_trials=2,      # small for test
        checkpoint_dir="./checkpoints",
        experiment_name="generic_classification_test",
    )

    print("\n=== Finished Classification Test ===")
    print("Best config:", result["best_config"])
    print("Best metric:", result["best_val_metric"])

