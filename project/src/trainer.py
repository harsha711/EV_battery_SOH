"""
trainer.py — Reusable Trainer class for all 4 SOH prediction models.

Wraps the standard training loop (Adam + ReduceLROnPlateau + early stopping)
into a fit() API so notebooks don't repeat boilerplate.

Usage:
    from src.trainer import Trainer

    trainer = Trainer(model, device)                    # most models
    trainer = Trainer(model, device, clip_grad=1.0)     # LSTM

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        eval_dataset=test_loader,   # treated same as val_loader
        num_epochs=30,
        learning_rate=1e-3,
        weight_decay=1e-5,
        patience=5,
        save_dir='../checkpoints',
    )
"""

import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Unified trainer for SOH regression models.

    Args:
        model:     Model to train (already moved to device).
        device:    torch.device to run on.
        clip_grad: If not None, clip gradient norm to this value each step.
                   Pass 1.0 for LSTM to stabilise 301-step sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        clip_grad: Optional[float] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.clip_grad = clip_grad

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        eval_dataset=None,          # accepted for API compatibility; treated same as val_loader
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        label_smoothing: float = 0.0,   # accepted but ignored — MSELoss has no smoothing
        save_dir: str = "../checkpoints",
    ) -> dict:
        """
        Train the model and return the history dict.

        Early stopping monitors validation MAE (SOH%). The best checkpoint is
        saved to save_dir/<ModelClass>_best.pt whenever a new best is found.

        Args:
            train_loader:    DataLoader for the training split.
            val_loader:      DataLoader for the validation/test split (used for
                             early stopping and per-epoch metrics).
            eval_dataset:    Ignored — kept for API compatibility with the
                             reference snippet. val_loader is used for all
                             evaluation.
            num_epochs:      Maximum number of training epochs.
            learning_rate:   Adam learning rate.
            weight_decay:    Adam L2 penalty.
            patience:        Early-stopping patience (epochs without improvement
                             on val MAE before stopping).
            label_smoothing: Accepted but not used (MSELoss regression task).
            save_dir:        Directory for the best-model checkpoint.

        Returns:
            history dict with lists: train_loss, val_mae, val_rmse, lr.
        """
        os.makedirs(save_dir, exist_ok=True)
        model_name = type(self.model).__name__
        ckpt_path = os.path.join(save_dir, f"{model_name}_best.pt")

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        history: dict[str, list] = {
            "train_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "lr": [],
        }

        best_val_mae = float("inf")
        no_improve = 0

        print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val MAE%':>10}  {'Val RMSE%':>11}  {'LR':>10}  {'Time':>6}")
        print("-" * 72)

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            val_mae, val_rmse = self._evaluate(val_loader)
            scheduler.step(val_mae)

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_mae"].append(val_mae)
            history["val_rmse"].append(val_rmse)
            history["lr"].append(current_lr)

            print(
                f"{epoch:>6}  {train_loss:>12.6f}  {val_mae:>10.4f}  {val_rmse:>11.4f}  "
                f"{current_lr:>10.2e}  {elapsed:>5.1f}s"
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                no_improve = 0
                torch.save(
                    {"epoch": epoch, "val_mae": val_mae, "model_state_dict": self.model.state_dict()},
                    ckpt_path,
                )
                print(f"         ✓ New best — checkpoint saved to {ckpt_path}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                    break

        print(f"\nBest val MAE: {best_val_mae:.4f}%  |  Checkpoint: {ckpt_path}")
        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """One training epoch. Returns mean loss over all batches."""
        self.model.train()
        total_loss = 0.0
        bar = tqdm(loader, desc="Training", leave=False, unit="batch")

        for x, y in bar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(self.model(x).squeeze(1), y)
            loss.backward()

            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)

            optimizer.step()
            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Evaluate on loader. Returns (MAE%, RMSE%) in SOH% units."""
        self.model.eval()
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        bar = tqdm(loader, desc="Validating", leave=False, unit="batch")

        for x, y in bar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            all_preds.append(self.model(x).squeeze(1).cpu())
            all_targets.append(y.cpu())

        preds = torch.cat(all_preds) * 100.0     # scale [0,1] → SOH%
        targets = torch.cat(all_targets) * 100.0

        mae = (preds - targets).abs().mean().item()
        rmse = math.sqrt(((preds - targets) ** 2).mean().item())

        return mae, rmse
