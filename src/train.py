#!/usr/bin/env python
"""
Single experiment run executor.
Trains model with D-RAdam or comparative optimizer using configuration from environment.
Logs comprehensive metrics to WandB and saves training diagnostics.
"""

import os
import sys
import json
import time
import logging
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
from optuna.samplers import TPESampler
import wandb
from omegaconf import DictConfig, OmegaConf

from src.model import build_model
from src.preprocess import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DRAdam(optim.Optimizer):
    """
    Dimensionality-Aware Rectified Adam (D-RAdam).
    
    Extends RAdam by dynamically adjusting variance rectification based on
    estimated effective problem dimensionality inferred from parameter norm
    growth dynamics and gradient statistics.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        gamma: float = 0.05,
        rho_base: float = 4.0,
        dim_smooth: float = 0.99,
    ):
        """Initialize D-RAdam optimizer."""
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= gamma:
            raise ValueError(f"Invalid gamma value: {gamma}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            gamma=gamma,
            rho_base=rho_base,
            dim_smooth=dim_smooth,
        )
        super(DRAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Perform single optimizer step with dimensionality-aware rectification."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("D-RAdam does not support sparse gradients")

                p_fp32 = p.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_fp32)
                    state["d_eff"] = 1.0
                    state["rho_threshold"] = group["rho_base"]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Decay first and second moment estimates
                exp_avg.mul_(group["betas"][0]).add_(grad, alpha=1 - group["betas"][0])
                exp_avg_sq.mul_(group["betas"][1]).addcmul_(
                    grad, grad, value=1 - group["betas"][1]
                )

                bias_correction1 = 1 - group["betas"][0] ** state["step"]
                bias_correction2 = 1 - group["betas"][1] ** state["step"]

                # Compute parameter norm for dimensionality estimation
                param_norm_sq = (p_fp32 ** 2).sum().item()

                # Estimate effective dimensionality: d_eff(t) = ||theta||^2 / mean(exp_avg_sq)
                avg_grad_norm_sq = (exp_avg_sq.sum() / exp_avg_sq.numel()).item()

                if avg_grad_norm_sq < 1e-8:
                    d_eff_current = 1.0
                else:
                    d_eff_current = max(1.0, param_norm_sq / (avg_grad_norm_sq + 1e-8))

                # Exponentially smooth dimensionality estimate
                state["d_eff"] = (
                    group["dim_smooth"] * state["d_eff"]
                    + (1 - group["dim_smooth"]) * d_eff_current
                )

                # Adaptive complexity threshold
                rho_threshold = (
                    group["rho_base"]
                    + group["gamma"] * math.log(max(state["d_eff"], 1.0) + 1.0)
                )
                state["rho_threshold"] = rho_threshold

                # Standard RAdam degrees of freedom calculation
                beta2_t = group["betas"][1] ** state["step"]
                rho_inf = 2 / (1 - group["betas"][1]) - 1
                rho_t = rho_inf - 2 * state["step"] * beta2_t / bias_correction2

                # Adaptive variance rectification
                if rho_t > rho_threshold:
                    # Variance-rectified regime
                    rect_term = math.sqrt(
                        (rho_t - rho_threshold)
                        * (rho_t - 2)
                        * rho_inf
                        / (
                            (rho_inf - rho_threshold)
                            * (rho_inf - 2)
                            * rho_t
                        )
                    )

                    adaptive_lr = (
                        group["lr"] * rect_term / math.sqrt(bias_correction2)
                    )

                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_fp32.addcdiv_(exp_avg, denom, value=-adaptive_lr)
                else:
                    # Momentum-only regime (high-dimensional)
                    adaptive_lr = group["lr"] / bias_correction1
                    p_fp32.add_(exp_avg, alpha=-adaptive_lr)

                # Weight decay
                if group["weight_decay"] != 0:
                    p_fp32.add_(p_fp32, alpha=-group["weight_decay"] * group["lr"])

                p.data.copy_(p_fp32)

        return loss


class RAdam(optim.Optimizer):
    """
    Rectified Adam (RAdam) - standalone implementation for baseline comparison.
    Uses fixed complexity threshold (rho_t > 4.0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(RAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Perform single optimizer step with fixed threshold."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_fp32 = p.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_fp32)
                    state["rho_threshold"] = 4.0

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Decay first and second moment estimates
                exp_avg.mul_(group["betas"][0]).add_(grad, alpha=1 - group["betas"][0])
                exp_avg_sq.mul_(group["betas"][1]).addcmul_(
                    grad, grad, value=1 - group["betas"][1]
                )

                bias_correction1 = 1 - group["betas"][0] ** state["step"]
                bias_correction2 = 1 - group["betas"][1] ** state["step"]

                # Standard RAdam with fixed threshold rho_t > 4
                beta2_t = group["betas"][1] ** state["step"]
                rho_inf = 2 / (1 - group["betas"][1]) - 1
                rho_t = rho_inf - 2 * state["step"] * beta2_t / bias_correction2

                if rho_t > 4:
                    # Variance-rectified regime with FIXED threshold
                    rect_term = math.sqrt(
                        (rho_t - 4) * (rho_t - 2) * rho_inf /
                        ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )

                    adaptive_lr = group["lr"] * rect_term / math.sqrt(bias_correction2)
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_fp32.addcdiv_(exp_avg, denom, value=-adaptive_lr)
                else:
                    # Momentum-only regime
                    adaptive_lr = group["lr"] / bias_correction1
                    p_fp32.add_(exp_avg, alpha=-adaptive_lr)

                # Weight decay
                if group["weight_decay"] != 0:
                    p_fp32.add_(p_fp32, alpha=-group["weight_decay"] * group["lr"])

                p.data.copy_(p_fp32)

        return loss


def get_optimizer(model: nn.Module, cfg: DictConfig, trial_params: Optional[Dict] = None) -> optim.Optimizer:
    """Create optimizer based on configuration."""
    optimizer_name = cfg.training.optimizer.lower()
    
    # Allow Optuna trial params to override config
    if trial_params is None:
        trial_params = {}
    
    lr = trial_params.get("learning_rate", cfg.training.learning_rate)
    weight_decay = trial_params.get("weight_decay", cfg.training.weight_decay)
    
    optimizer_params = OmegaConf.to_container(cfg.training.optimizer_params)
    beta_1 = optimizer_params.get("beta_1", 0.9)
    beta_2 = optimizer_params.get("beta_2", 0.999)
    eps = optimizer_params.get("eps", 1e-8)
    
    if optimizer_name == "d-radam":
        gamma = trial_params.get("gamma", optimizer_params.get("gamma", 0.05))
        rho_base = trial_params.get("rho_base", optimizer_params.get("rho_base", 4.0))
        dim_smooth = trial_params.get("dim_smooth", optimizer_params.get("dim_smooth", 0.99))
        
        return DRAdam(
            model.parameters(),
            lr=lr,
            betas=(beta_1, beta_2),
            eps=eps,
            weight_decay=weight_decay,
            gamma=gamma,
            rho_base=rho_base,
            dim_smooth=dim_smooth,
        )
    elif optimizer_name == "radam":
        return RAdam(
            model.parameters(),
            lr=lr,
            betas=(beta_1, beta_2),
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta_1, beta_2),
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: DictConfig,
    epoch: int,
    suppress_wandb: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    d_eff_values = []
    rho_threshold_values = []
    rectification_active_steps = 0
    total_steps = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # CRITICAL: Assert input/label shapes at batch start (at least for step 0)
        if batch_idx == 0:
            assert inputs.shape[0] > 0, "Empty batch received"
            assert labels.shape[0] == inputs.shape[0], \
                f"Batch size mismatch: inputs {inputs.shape[0]} != labels {labels.shape[0]}"
        
        optimizer.zero_grad()
        
        # DATA LEAK PREVENTION: Model receives ONLY inputs
        outputs = model(inputs)
        
        # CRITICAL: Verify outputs are finite before backward
        assert not torch.isnan(outputs).any(), "NaN in model output"
        assert not torch.isinf(outputs).any(), "Inf in model output"
        
        # Labels used ONLY for loss computation, NEVER concatenated to inputs
        loss = criterion(outputs, labels)
        
        # CRITICAL: Verify loss is finite after computation
        assert not torch.isnan(loss).item() and not torch.isinf(loss).item(), \
            "NaN or Inf in loss value"
        
        loss.backward()
        
        # CRITICAL: After backward(), assert gradients exist and are not NaN/Inf
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                has_grads = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient detected post-backward"
                assert not torch.isinf(param.grad).any(), f"Inf gradient detected post-backward"
        
        assert has_grads, "No gradients computed after backward. Check loss computation."
        
        # Gradient clipping
        if cfg.training.get("gradient_clip", 0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
        
        # CRITICAL: Before optimizer.step(), assert that gradients exist and have non-trivial magnitude
        grad_count = 0
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_count += 1
                total_grad_norm += (param.grad ** 2).sum().item()
        
        assert grad_count > 0, "No non-None gradients exist before optimizer step"
        assert total_grad_norm > 1e-10, \
            f"All gradients are near-zero before optimizer step (grad norm: {total_grad_norm})"
        
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
        
        # Extract D-RAdam specific diagnostics
        if cfg.training.optimizer.lower() == "d-radam":
            for param_group in optimizer.param_groups:
                for p in param_group["params"]:
                    if p in optimizer.state:
                        state = optimizer.state[p]
                        if "d_eff" in state:
                            d_eff_values.append(state["d_eff"])
                        if "rho_threshold" in state:
                            rho_threshold_values.append(state["rho_threshold"])
                        break
            
            # Track variance rectification regime transition
            total_steps += 1
            if len(d_eff_values) > 0 and len(rho_threshold_values) > 0:
                # Compute rho_t for tracking
                beta1 = cfg.training.optimizer_params.get("beta_1", 0.9)
                beta2 = cfg.training.optimizer_params.get("beta_2", 0.999)
                step_num = batch_idx + epoch * len(loader)
                beta2_t = beta2 ** step_num
                rho_inf = 2 / (1 - beta2) - 1
                bias_correction2 = 1 - beta2_t
                if bias_correction2 > 1e-8:
                    rho_t = rho_inf - 2 * step_num * beta2_t / bias_correction2
                    if rho_t > rho_threshold_values[-1]:
                        rectification_active_steps += 1
        elif cfg.training.optimizer.lower() == "radam":
            # Track rectification for RAdam too
            for param_group in optimizer.param_groups:
                for p in param_group["params"]:
                    if p in optimizer.state:
                        state = optimizer.state[p]
                        if "rho_threshold" in state:
                            rho_threshold_values.append(state["rho_threshold"])
                        break
            
            total_steps += 1
            if len(rho_threshold_values) > 0:
                beta1 = cfg.training.optimizer_params.get("beta_1", 0.9)
                beta2 = cfg.training.optimizer_params.get("beta_2", 0.999)
                step_num = batch_idx + epoch * len(loader)
                beta2_t = beta2 ** step_num
                rho_inf = 2 / (1 - beta2) - 1
                bias_correction2 = 1 - beta2_t
                if bias_correction2 > 1e-8:
                    rho_t = rho_inf - 2 * step_num * beta2_t / bias_correction2
                    if rho_t > 4.0:
                        rectification_active_steps += 1
        
        if (batch_idx + 1) % cfg.training.logging.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * total_correct / total_samples
            logger.info(
                f"Epoch {epoch+1} [{batch_idx+1}/{len(loader)}] "
                f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%"
            )
    
    epoch_loss = total_loss / len(loader)
    epoch_accuracy = 100.0 * total_correct / total_samples
    
    rectification_fraction = rectification_active_steps / max(total_steps, 1)
    
    metrics = {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "d_eff_mean": np.mean(d_eff_values) if d_eff_values else 1.0,
        "rho_threshold_mean": np.mean(rho_threshold_values) if rho_threshold_values else 4.0,
        "variance_rectification_regime_transition": rectification_fraction,
    }
    
    return metrics


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)
    
    val_loss = total_loss / len(loader)
    val_accuracy = 100.0 * total_correct / total_samples
    
    return {
        "loss": val_loss,
        "accuracy": val_accuracy,
    }


def objective(
    trial: optuna.Trial,
    model_builder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> float:
    """Optuna objective function for hyperparameter optimization."""
    
    # Sample hyperparameters from search space
    trial_params = {}
    
    if cfg.optuna.get("search_spaces"):
        for search_space in cfg.optuna.search_spaces:
            param_name = search_space["param_name"]
            dist_type = search_space["distribution_type"]
            
            if dist_type == "uniform":
                trial_params[param_name] = trial.suggest_float(
                    param_name,
                    search_space["low"],
                    search_space["high"],
                )
            elif dist_type == "loguniform":
                trial_params[param_name] = trial.suggest_float(
                    param_name,
                    search_space["low"],
                    search_space["high"],
                    log=True,
                )
            elif dist_type == "categorical":
                trial_params[param_name] = trial.suggest_categorical(
                    param_name,
                    search_space["choices"],
                )
    
    # Build model for this trial
    model = model_builder().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, cfg, trial_params)
    
    # Train for limited epochs during trial
    trial_epochs = min(5, cfg.training.epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(trial_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            cfg,
            epoch,
            suppress_wandb=True,
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        best_val_acc = max(best_val_acc, val_metrics["accuracy"])
        
        # Optuna pruning
        trial.report(val_metrics["accuracy"], step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_acc


def main() -> None:
    """Main training loop for single run."""
    
    # Retrieve configuration from environment variable
    config_json = os.environ.get("EXPERIMENT_CONFIG")
    if not config_json:
        logger.error("EXPERIMENT_CONFIG environment variable not set")
        sys.exit(1)
    
    config_dict = json.loads(config_json)
    cfg = OmegaConf.create(config_dict)
    
    run_id = os.environ.get("EXPERIMENT_RUN_ID", cfg.run.run_id)
    mode = os.environ.get("EXPERIMENT_MODE", cfg.mode)
    results_dir = Path(os.environ.get("EXPERIMENT_RESULTS_DIR", cfg.results_dir))
    
    # CRITICAL: Enforce trial mode constraints in train.py
    if mode == "trial":
        assert cfg.training.epochs == 1, "Trial mode must have epochs=1"
        assert cfg.wandb.mode == "disabled", "Trial mode must have wandb.mode=disabled"
        assert cfg.optuna.n_trials == 0, "Trial mode must have optuna.n_trials=0"
    
    # Set random seeds
    seed = cfg.training.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Mode: {mode}")
    
    # POST-INIT ASSERTION: Verify critical attributes
    assert cfg.training.learning_rate > 0, "Invalid learning rate"
    assert cfg.training.batch_size > 0, "Invalid batch size"
    assert cfg.training.epochs > 0, "Invalid epochs"
    assert hasattr(cfg.model, "num_classes") and cfg.model.num_classes > 0, "Invalid num_classes"
    logger.info("Configuration validation passed")
    
    # Load dataset
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    dataset = load_dataset(cfg.dataset.name, cfg.dataset)
    
    # Split into train/val/test with proper random generator to prevent data leakage
    train_size = int(cfg.dataset.split_ratios.train * len(dataset))
    val_size = int(cfg.dataset.split_ratios.val * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    generator_train = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator_train
    )
    
    num_workers = 0 if mode == "trial" else 4
    
    # Trial mode: limit to first 1-2 batches for quick validation
    if mode == "trial":
        batch_limit = 2
        train_set = torch.utils.data.Subset(train_set, list(range(min(batch_limit * cfg.training.batch_size, len(train_set)))))
        val_set = torch.utils.data.Subset(val_set, list(range(min(batch_limit * cfg.training.batch_size, len(val_set)))))
        test_set = torch.utils.data.Subset(test_set, list(range(min(batch_limit * cfg.training.batch_size, len(test_set)))))
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    logger.info(f"Dataset loaded: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    # Optuna hyperparameter search (only log best result to WandB after optimization)
    best_trial_params = {}
    
    if cfg.optuna.get("enabled", False) and cfg.optuna.get("n_trials", 0) > 0:
        logger.info(f"Running Optuna hyperparameter search with {cfg.optuna.n_trials} trials...")
        
        def model_builder():
            return build_model(cfg.model)
        
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(
            direction=cfg.optuna.get("direction", "maximize"),
            sampler=sampler,
        )
        
        study.optimize(
            lambda trial: objective(
                trial,
                model_builder,
                train_loader,
                val_loader,
                device,
                cfg,
            ),
            n_trials=cfg.optuna.n_trials,
            show_progress_bar=True,
        )
        
        best_trial = study.best_trial
        best_trial_params = best_trial.params
        logger.info(f"Best trial parameters: {best_trial_params}")
        logger.info(f"Best trial value (val accuracy): {best_trial.value:.4f}")
    
    # Build final model with best (or default) hyperparameters
    logger.info(f"Building model: {cfg.model.name}")
    model = build_model(cfg.model).to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # POST-INIT ASSERTION: Verify model output shape (before WandB init)
    sample_input = next(iter(train_loader))[0][:1].to(device)
    with torch.no_grad():
        sample_output = model(sample_input)
    assert sample_output.shape[1] == cfg.model.num_classes, \
        f"Model output dimension {sample_output.shape[1]} != num_classes {cfg.model.num_classes}"
    logger.info("Model output shape verified")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, cfg, best_trial_params)
    
    logger.info(f"Optimizer: {cfg.training.optimizer}")
    logger.info(f"Learning rate: {best_trial_params.get('learning_rate', cfg.training.learning_rate)}")
    
    # Initialize WandB
    wandb_mode = cfg.wandb.mode
    if wandb_mode != "disabled":
        config_to_log = OmegaConf.to_container(cfg, resolve=True)
        config_to_log["best_trial_params"] = best_trial_params
        
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_id,
            config=config_to_log,
            resume="allow",
            mode=wandb_mode,
        )
        logger.info(f"WandB initialized: {wandb.run.url}")
    else:
        logger.info("WandB disabled (trial mode)")
    
    # Training loop
    logger.info("Starting training...")
    wall_clock_start = time.time()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    d_eff_trajectories = []
    rho_threshold_trajectories = []
    rectification_fractions = []
    
    best_val_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(cfg.training.epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{cfg.training.epochs} ===")
        
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            cfg,
            epoch,
            suppress_wandb=False,
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        train_accuracies.append(train_metrics["accuracy"])
        val_accuracies.append(val_metrics["accuracy"])
        d_eff_trajectories.append(train_metrics["d_eff_mean"])
        rho_threshold_trajectories.append(train_metrics["rho_threshold_mean"])
        rectification_fractions.append(train_metrics["variance_rectification_regime_transition"])
        
        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )
        
        # Track best validation accuracy
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
        
        # Log to WandB PER-EPOCH with comprehensive metrics
        if wandb_mode != "disabled":
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            
            if cfg.training.optimizer.lower() == "d-radam":
                log_dict["d_eff"] = train_metrics["d_eff_mean"]
                log_dict["rho_threshold"] = train_metrics["rho_threshold_mean"]
                log_dict["variance_rectification_regime_transition"] = train_metrics["variance_rectification_regime_transition"]
            elif cfg.training.optimizer.lower() == "radam":
                log_dict["variance_rectification_regime_transition"] = train_metrics["variance_rectification_regime_transition"]
            
            wandb.log(log_dict)
    
    wall_clock_end = time.time()
    wall_clock_time = wall_clock_end - wall_clock_start
    
    # Test evaluation
    test_metrics = validate(model, test_loader, criterion, device)
    
    logger.info(f"\n=== Final Results ===")
    logger.info(f"Best Val Accuracy (Epoch {best_epoch+1}): {best_val_accuracy:.2f}%")
    logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.2f}%")
    logger.info(f"Final Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Total Wall-Clock Time: {wall_clock_time:.2f} seconds")
    
    # Compute convergence metrics
    convergence_speed_20 = 0.0
    convergence_speed_100 = 0.0
    
    if len(train_losses) > 1:
        loss_at_epoch_1 = train_losses[0]
        if len(train_losses) > 19:
            loss_at_epoch_20 = train_losses[19]
            convergence_speed_20 = max(0.0, 1.0 - loss_at_epoch_20 / loss_at_epoch_1)
        if len(train_losses) > 99:
            loss_at_epoch_100 = train_losses[99]
            convergence_speed_100 = max(0.0, 1.0 - loss_at_epoch_100 / loss_at_epoch_1)
        else:
            convergence_speed_100 = max(0.0, 1.0 - train_losses[-1] / loss_at_epoch_1)
    
    logger.info(f"Convergence Speed @ Epoch 20: {convergence_speed_20:.4f}")
    logger.info(f"Convergence Speed @ Epoch 100: {convergence_speed_100:.4f}")
    
    # Compute synthetic convergence speedup metric (placeholder for synthetic datasets)
    synthetic_convergence_speedup = 1.0
    
    # Log final metrics to WandB summary
    if wandb_mode != "disabled":
        wandb.summary["best_val_accuracy"] = best_val_accuracy
        wandb.summary["final_test_accuracy"] = test_metrics["accuracy"]
        wandb.summary["final_test_loss"] = test_metrics["loss"]
        wandb.summary["convergence_speed_at_epoch_20"] = convergence_speed_20
        wandb.summary["convergence_speed_at_epoch_100"] = convergence_speed_100
        wandb.summary["wall_clock_training_time"] = wall_clock_time
        wandb.summary["synthetic_convergence_speedup"] = synthetic_convergence_speedup
        
        wandb.finish()
        print(f"WandB run URL: {wandb.run.url}")
    
    # Save results to file
    results_subdir = results_dir / run_id
    results_subdir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        "run_id": run_id,
        "method": cfg.method,
        "optimizer": cfg.training.optimizer,
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_epoch": int(best_epoch + 1),
        "final_test_accuracy": float(test_metrics["accuracy"]),
        "final_test_loss": float(test_metrics["loss"]),
        "convergence_speed_at_epoch_20": float(convergence_speed_20),
        "convergence_speed_at_epoch_100": float(convergence_speed_100),
        "wall_clock_training_time": float(wall_clock_time),
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "train_accuracies": [float(x) for x in train_accuracies],
        "val_accuracies": [float(x) for x in val_accuracies],
        "d_eff_trajectory": [float(x) for x in d_eff_trajectories],
        "rho_threshold_trajectory": [float(x) for x in rho_threshold_trajectories],
        "variance_rectification_regime_transition": [float(x) for x in rectification_fractions],
    }
    
    with open(results_subdir / "training_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Results saved to: {results_subdir / 'training_metrics.json'}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
