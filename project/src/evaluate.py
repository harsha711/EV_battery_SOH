"""
evaluate.py — Evaluation and comparison of all 4 SOH prediction models.

Loads saved checkpoints, runs inference on the test set, computes MAE and
RMSE for each model, and generates comparison figures.

Generated figures:
    1. metrics_comparison.png   — Bar chart: MAE and RMSE per model
    2. scatter_all_models.png   — 4-panel scatter: pred vs actual SOH
    3. training_curves.png      — Val MAE over epochs for all 4 models
    4. error_hist_{model}.png   — Error histogram for the best model
    5. attn_viz.png             — Transformer attention weights (mean over samples)
"""

import json
import math
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import get_dataloaders
from src.models import MODEL_REGISTRY
from src.models.transformer import SOHTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "mlp": "MLP Baseline",
    "cnn": "1D-CNN",
    "lstm": "LSTM",
    "transformer": "Transformer",
}
MODEL_COLORS: dict[str, str] = {
    "mlp": "#4C72B0",
    "cnn": "#DD8452",
    "lstm": "#55A868",
    "transformer": "#C44E52",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    model_name: str,
    device: torch.device,
) -> nn.Module:
    """
    Instantiate a model from the registry and load its checkpoint weights.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model_name: Key into MODEL_REGISTRY (e.g. 'mlp', 'cnn').
        device: torch device to map weights to.

    Returns:
        Model with loaded weights, in eval mode.
    """
    model = MODEL_REGISTRY[model_name]().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {model_name} from epoch {ckpt.get('epoch', '?')} "
          f"(val MAE {ckpt.get('val_mae', float('nan')):.4f}%)")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run inference and compute MAE and RMSE in SOH% units.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader (test set).
        device: torch device.

    Returns:
        Dict with keys:
            'mae':         float (SOH%)
            'rmse':        float (SOH%)
            'predictions': np.ndarray (N,) in SOH%
            'targets':     np.ndarray (N,) in SOH%
    """
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    model.eval()
    for x, y in loader:
        x = x.to(device)
        pred = model(x).squeeze(1).cpu()
        all_preds.append(pred)
        all_targets.append(y)

    preds = torch.cat(all_preds).numpy() * 100.0     # → SOH%
    targets = torch.cat(all_targets).numpy() * 100.0

    mae = float(np.abs(preds - targets).mean())
    rmse = float(math.sqrt(((preds - targets) ** 2).mean()))

    return {
        "mae": mae,
        "rmse": rmse,
        "predictions": preds,
        "targets": targets,
    }


# ---------------------------------------------------------------------------
# Individual model evaluation
# ---------------------------------------------------------------------------


def evaluate_single(
    model_name: str,
    checkpoint_path: str,
    h5_path: str,
    stats_path: str,
    output_dir: str,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """
    Evaluate one model checkpoint on the test set.

    Prints MAE and RMSE, saves a scatter plot.

    Args:
        model_name: Key into MODEL_REGISTRY.
        checkpoint_path: Path to .pt checkpoint.
        h5_path: Path to sequences.h5.
        stats_path: Path to norm_stats.json.
        output_dir: Root output directory.
        device: torch device.
        batch_size: Batch size for inference.

    Returns:
        Dict with 'mae', 'rmse', 'predictions', 'targets'.
    """
    _, test_loader = get_dataloaders(
        h5_path=h5_path,
        stats_path=stats_path,
        batch_size=batch_size,
        num_workers=0,
    )
    model = load_model(checkpoint_path, model_name, device)
    results = run_inference(model, test_loader, device)

    print(f"\n{MODEL_DISPLAY_NAMES[model_name]}:")
    print(f"  MAE:  {results['mae']:.4f} SOH%")
    print(f"  RMSE: {results['rmse']:.4f} SOH%")

    # Save scatter plot
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _plot_scatter_single(
        results["predictions"],
        results["targets"],
        model_name,
        os.path.join(fig_dir, f"scatter_{model_name}.png"),
    )

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_scatter_single(
    preds: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    save_path: str,
) -> None:
    """Scatter plot of predicted vs actual SOH for one model."""
    mae = float(np.abs(preds - targets).mean())
    rmse = float(math.sqrt(((preds - targets) ** 2).mean()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(targets, preds, alpha=0.3, s=6, color=MODEL_COLORS.get(model_name, "steelblue"))

    lims = [min(targets.min(), preds.min()) - 2, max(targets.max(), preds.max()) + 2]
    ax.plot(lims, lims, "k--", lw=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual SOH (%)")
    ax.set_ylabel("Predicted SOH (%)")
    ax.set_title(f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}\nMAE={mae:.2f}%  RMSE={rmse:.2f}%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved scatter plot: {save_path}")


def _plot_metrics_comparison(
    all_results: dict[str, dict],
    save_path: str,
) -> None:
    """Bar chart comparing MAE and RMSE across all 4 models."""
    model_names = list(all_results.keys())
    maes = [all_results[m]["mae"] for m in model_names]
    rmses = [all_results[m]["rmse"] for m in model_names]
    labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in model_names]
    colors = [MODEL_COLORS.get(m, "gray") for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars1 = ax1.bar(x, maes, width=0.6, color=colors, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("MAE (SOH%)")
    ax1.set_title("MAE Comparison")
    ax1.bar_label(bars1, fmt="%.2f", padding=3, fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(x, rmses, width=0.6, color=colors, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("RMSE (SOH%)")
    ax2.set_title("RMSE Comparison")
    ax2.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Comparison — Test Set Performance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved metrics comparison: {save_path}")


def _plot_scatter_grid(
    all_results: dict[str, dict],
    save_path: str,
) -> None:
    """4-panel scatter plot: predicted vs actual for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, (model_name, results) in zip(axes, all_results.items()):
        preds = results["predictions"]
        targets = results["targets"]
        mae = results["mae"]
        rmse = results["rmse"]

        ax.scatter(targets, preds, alpha=0.25, s=5, color=MODEL_COLORS.get(model_name, "gray"))
        lims = [
            min(targets.min(), preds.min()) - 2,
            max(targets.max(), preds.max()) + 2,
        ]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual SOH (%)", fontsize=9)
        ax.set_ylabel("Predicted SOH (%)", fontsize=9)
        ax.set_title(
            f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}\n"
            f"MAE={mae:.2f}%  RMSE={rmse:.2f}%",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle("Predicted vs Actual SOH — All Models", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter grid: {save_path}")


def _plot_training_curves(
    metrics_dir: str,
    save_path: str,
) -> None:
    """Overlay val MAE training curves for all 4 models on one plot."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name in MODEL_REGISTRY.keys():
        history_path = os.path.join(metrics_dir, f"{model_name}_history.json")
        if not os.path.exists(history_path):
            print(f"  Warning: no history found for {model_name}, skipping curve.")
            continue
        with open(history_path, "r") as fp:
            history = json.load(fp)
        val_mae = history.get("val_mae", [])
        epochs = range(1, len(val_mae) + 1)
        ax.plot(
            epochs,
            val_mae,
            label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
            color=MODEL_COLORS.get(model_name, "gray"),
            linewidth=1.8,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (SOH%)")
    ax.set_title("Training Curves — Validation MAE", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves: {save_path}")


def _plot_error_histogram(
    results: dict,
    model_name: str,
    save_path: str,
) -> None:
    """Histogram of prediction errors (pred - actual) for the best model."""
    errors = results["predictions"] - results["targets"]
    mae = results["mae"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=60, color=MODEL_COLORS.get(model_name, "steelblue"), edgecolor="white",
            alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", lw=1.2, label="Zero error")
    ax.axvline(errors.mean(), color="red", linestyle="-", lw=1.5,
               label=f"Mean={errors.mean():.2f}%")
    ax.set_xlabel("Prediction Error (SOH%)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Error Distribution — {MODEL_DISPLAY_NAMES.get(model_name, model_name)}\n"
        f"MAE={mae:.2f}%  std={errors.std():.2f}%",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved error histogram: {save_path}")


def _plot_attention(
    checkpoint_path: str,
    h5_path: str,
    stats_path: str,
    save_path: str,
    device: torch.device,
    n_samples: int = 8,
) -> None:
    """
    Visualize Transformer attention weights averaged over samples.

    Loads a few test samples, extracts per-layer attention weights,
    and plots the mean attention pattern across sequence positions.

    Args:
        checkpoint_path: Path to the transformer checkpoint.
        h5_path: Path to sequences.h5.
        stats_path: Path to norm_stats.json.
        save_path: Output figure path.
        device: torch device.
        n_samples: Number of test samples to average over.
    """
    model = load_model(checkpoint_path, "transformer", device)
    assert isinstance(model, SOHTransformer)

    _, test_loader = get_dataloaders(
        h5_path=h5_path, stats_path=stats_path, batch_size=n_samples, num_workers=0
    )
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch[:n_samples].to(device)

    attn_per_layer = model.get_attention_weights(x_batch)  # list of (batch, heads, 301, 301)

    n_layers = len(attn_per_layer)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for i, (ax, attn) in enumerate(zip(axes, attn_per_layer)):
        # Mean over batch and heads: (301, 301)
        attn_mean = attn.mean(dim=(0, 1)).numpy()
        im = ax.imshow(attn_mean, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_title(f"Layer {i+1}", fontsize=10)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Transformer Attention Weights (mean over {n_samples} samples)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved attention visualization: {save_path}")


# ---------------------------------------------------------------------------
# Compare all models
# ---------------------------------------------------------------------------


def compare_all_models(
    results_dir: str,
    h5_path: str,
    stats_path: str,
    device: torch.device,
    batch_size: int = 256,
) -> None:
    """
    Load all 4 model checkpoints, run inference, and generate all comparison figures.

    Skips models whose checkpoints don't exist yet with a warning.

    Args:
        results_dir: Root results directory (contains checkpoints/, metrics/, figures/).
        h5_path: Path to sequences.h5.
        stats_path: Path to norm_stats.json.
        device: torch device.
        batch_size: Inference batch size.
    """
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    metrics_dir = os.path.join(results_dir, "metrics")
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    _, test_loader = get_dataloaders(
        h5_path=h5_path, stats_path=stats_path, batch_size=batch_size, num_workers=0
    )

    all_results: dict[str, dict] = {}

    for model_name in MODEL_REGISTRY.keys():
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"Warning: checkpoint not found for {model_name} ({ckpt_path}), skipping.")
            continue

        print(f"\nEvaluating {MODEL_DISPLAY_NAMES[model_name]}...")
        model = load_model(ckpt_path, model_name, device)
        results = run_inference(model, test_loader, device)
        all_results[model_name] = results
        print(f"  MAE:  {results['mae']:.4f} SOH%")
        print(f"  RMSE: {results['rmse']:.4f} SOH%")

    if not all_results:
        print("No model checkpoints found. Train at least one model first.")
        return

    # -----------------------------------------------------------------------
    # 1. Metrics bar chart
    # -----------------------------------------------------------------------
    _plot_metrics_comparison(all_results, os.path.join(fig_dir, "metrics_comparison.png"))

    # -----------------------------------------------------------------------
    # 2. Scatter grid
    # -----------------------------------------------------------------------
    if len(all_results) >= 2:
        _plot_scatter_grid(all_results, os.path.join(fig_dir, "scatter_all_models.png"))

    # -----------------------------------------------------------------------
    # 3. Training curves
    # -----------------------------------------------------------------------
    _plot_training_curves(metrics_dir, os.path.join(fig_dir, "training_curves.png"))

    # -----------------------------------------------------------------------
    # 4. Error histogram for the best model
    # -----------------------------------------------------------------------
    best_model = min(all_results, key=lambda m: all_results[m]["mae"])
    print(f"\nBest model by MAE: {MODEL_DISPLAY_NAMES[best_model]} "
          f"(MAE={all_results[best_model]['mae']:.4f}%)")
    _plot_error_histogram(
        all_results[best_model],
        best_model,
        os.path.join(fig_dir, f"error_hist_{best_model}.png"),
    )

    # -----------------------------------------------------------------------
    # 5. Transformer attention (if available)
    # -----------------------------------------------------------------------
    transformer_ckpt = os.path.join(ckpt_dir, "transformer_best.pt")
    if os.path.exists(transformer_ckpt):
        print("\nGenerating attention visualization...")
        try:
            _plot_attention(
                transformer_ckpt,
                h5_path,
                stats_path,
                os.path.join(fig_dir, "attn_viz.png"),
                device=device,
            )
        except Exception as e:
            print(f"  Warning: attention visualization failed: {e}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"{'Model':<20}  {'MAE (SOH%)':<12}  {'RMSE (SOH%)':<12}")
    print("-" * 55)
    for model_name in MODEL_REGISTRY.keys():
        if model_name not in all_results:
            continue
        r = all_results[model_name]
        name = MODEL_DISPLAY_NAMES[model_name]
        print(f"{name:<20}  {r['mae']:<12.4f}  {r['rmse']:<12.4f}")
    print(f"{'='*55}")
    print(f"\nAll figures saved to {fig_dir}/")


