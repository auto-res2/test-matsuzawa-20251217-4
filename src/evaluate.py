#!/usr/bin/env python
"""
Independent evaluation and visualization script for experiment results.
Retrieves comprehensive data from WandB API and generates comparison figures.
Executed as separate workflow after all training runs complete.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for evaluate.py."""
    parser = argparse.ArgumentParser(
        description="Evaluate D-RAdam experiment results from WandB"
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Arguments in key=value format or positional",
    )

    args = parser.parse_args()

    # Parse key=value style arguments
    parsed = {}
    positional = []

    for arg in args.args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key] = value
        else:
            positional.append(arg)

    # Handle positional arguments (backwards compatibility)
    if len(positional) >= 2:
        parsed["results_dir"] = positional[0]
        parsed["run_ids"] = positional[1]

    # Validate required arguments
    if "results_dir" not in parsed:
        parser.error("results_dir is required")
    if "run_ids" not in parsed:
        parser.error("run_ids is required")

    # Create namespace object
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    return Args(**parsed)


def load_wandb_config() -> Dict:
    """Load WandB configuration from config/config.yaml."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file required at: {config_path}\n"
            "Please ensure config/config.yaml exists with wandb settings."
        )
    
    cfg = OmegaConf.load(config_path)
    if not hasattr(cfg, 'wandb'):
        raise ValueError("config/config.yaml must contain 'wandb' section")
    
    return OmegaConf.to_container(cfg.wandb)


def retrieve_run_data(entity: str, project: str, run_id: str, results_dir: Path) -> Optional[Dict]:
    """Retrieve comprehensive run data from local files or WandB API."""
    logger.info(f"Retrieving data for run: {run_id}")

    # First, try to load from local training_metrics.json
    local_metrics_path = results_dir / run_id / "training_metrics.json"
    if local_metrics_path.exists():
        logger.info(f"Loading data from local file: {local_metrics_path}")
        try:
            with open(local_metrics_path, "r") as f:
                local_data = json.load(f)

            # Convert local data format to expected format
            # Create history DataFrame from trajectory data
            history_dict = {}

            if "train_losses" in local_data and local_data["train_losses"]:
                history_dict["train_loss"] = local_data["train_losses"]
            if "val_losses" in local_data and local_data["val_losses"]:
                history_dict["val_loss"] = local_data["val_losses"]
            if "train_accuracies" in local_data and local_data["train_accuracies"]:
                history_dict["train_accuracy"] = local_data["train_accuracies"]
            if "val_accuracies" in local_data and local_data["val_accuracies"]:
                history_dict["val_accuracy"] = local_data["val_accuracies"]
            if "d_eff_trajectory" in local_data and local_data["d_eff_trajectory"]:
                history_dict["d_eff"] = local_data["d_eff_trajectory"]
            if "rho_threshold_trajectory" in local_data and local_data["rho_threshold_trajectory"]:
                history_dict["rho_threshold"] = local_data["rho_threshold_trajectory"]
            if "variance_rectification_regime_transition" in local_data and local_data["variance_rectification_regime_transition"]:
                history_dict["variance_rectification_regime_transition"] = local_data["variance_rectification_regime_transition"]

            # Make sure all arrays have the same length
            if history_dict:
                max_len = max(len(v) for v in history_dict.values())
                history_dict["epoch"] = list(range(1, max_len + 1))
            else:
                history_dict["epoch"] = []

            history = pd.DataFrame(history_dict)

            # Create summary from local data
            summary = {
                "final_test_accuracy": local_data.get("final_test_accuracy", 0.0),
                "final_test_loss": local_data.get("final_test_loss", 0.0),
                "convergence_speed_at_epoch_20": local_data.get("convergence_speed_at_epoch_20", 0.0),
                "convergence_speed_at_epoch_100": local_data.get("convergence_speed_at_epoch_100", 0.0),
                "wall_clock_training_time": local_data.get("wall_clock_training_time", 0.0),
                "best_val_accuracy": local_data.get("best_val_accuracy", 0.0),
                "best_val_epoch": local_data.get("best_val_epoch", 0),
            }

            # Create config from local data
            config = {
                "method": local_data.get("method", ""),
                "optimizer": local_data.get("optimizer", ""),
            }

            return {
                "history": history,
                "summary": summary,
                "config": config,
                "run_id": run_id,
            }
        except Exception as e:
            logger.error(f"Error loading local data for run {run_id}: {e}")
            # Fall through to WandB API

    # Fall back to WandB API if local file doesn't exist
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")

        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)

        return {
            "history": history,
            "summary": summary,
            "config": config,
            "run_id": run_id,
        }
    except Exception as e:
        logger.error(f"Error retrieving data for run {run_id}: {e}")
        return None


def export_per_run_metrics(run_data: Dict, results_dir: Path) -> None:
    """Export per-run metrics to JSON file."""
    run_id = run_data["run_id"]
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dict = {
        "run_id": run_id,
        "summary": run_data["summary"],
        "config": run_data["config"],
    }
    
    history = run_data["history"]
    if not history.empty:
        metrics_dict["history_summary"] = {
            "final_epoch": int(history.iloc[-1]["epoch"]) if "epoch" in history.columns else len(history),
            "final_train_loss": float(history.iloc[-1]["train_loss"]) if "train_loss" in history.columns else None,
            "final_val_loss": float(history.iloc[-1]["val_loss"]) if "val_loss" in history.columns else None,
            "final_train_accuracy": float(history.iloc[-1]["train_accuracy"]) if "train_accuracy" in history.columns else None,
            "final_val_accuracy": float(history.iloc[-1]["val_accuracy"]) if "val_accuracy" in history.columns else None,
        }
    
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    logger.info(f"Saved metrics: {metrics_path}")
    print(f"  - {metrics_path}")


def generate_per_run_figures(run_data: Dict, results_dir: Path) -> None:
    """Generate per-run comparison figures."""
    run_id = run_data["run_id"]
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    history = run_data["history"]
    if history.empty:
        logger.warning(f"No history data for run {run_id}")
        return

    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
    })

    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "epoch" in history.columns:
        epochs = history["epoch"].values
    else:
        epochs = np.arange(1, len(history) + 1)

    # Loss curve
    if "train_loss" in history.columns:
        axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=5, linewidth=2)
    if "val_loss" in history.columns:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=5, linewidth=2)

    # Mark epoch 20 if available
    if len(epochs) > 19:
        axes[0].axvline(x=epochs[19], color="red", linestyle="--", alpha=0.5, label="Epoch 20", linewidth=1.5)
        if "train_loss" in history.columns:
            loss_20 = history["train_loss"].iloc[19]
            axes[0].text(epochs[19], loss_20, f" {loss_20:.3f}", fontsize=10)

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(f"{run_id} - Loss Trajectory", fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linewidth=0.5)

    # For single epoch data, adjust x-axis to show clearly
    if len(epochs) == 1:
        axes[0].set_xlim([0.5, 1.5])

    # Accuracy curve
    if "train_accuracy" in history.columns:
        axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy", marker="o", markersize=5, linewidth=2)
    if "val_accuracy" in history.columns:
        axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy", marker="s", markersize=5, linewidth=2)

    # Mark epoch 20 if available
    if len(epochs) > 19:
        axes[1].axvline(x=epochs[19], color="red", linestyle="--", alpha=0.5, label="Epoch 20", linewidth=1.5)
        if "val_accuracy" in history.columns:
            acc_20 = history["val_accuracy"].iloc[19]
            axes[1].text(epochs[19], acc_20, f" {acc_20:.1f}", fontsize=10)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title(f"{run_id} - Accuracy Trajectory", fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linewidth=0.5)

    # For single epoch data, adjust x-axis to show clearly
    if len(epochs) == 1:
        axes[1].set_xlim([0.5, 1.5])

    plt.tight_layout()

    learning_curve_path = run_dir / f"{run_id}_learning_curve_train_val.pdf"
    plt.savefig(learning_curve_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved learning curves: {learning_curve_path}")
    print(f"  - {learning_curve_path}")
    
    # Dimensionality trajectory (D-RAdam only)
    if "d_eff" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, history["d_eff"], label="Effective Dimensionality d_eff(t)", marker="o", markersize=5, linewidth=2, color='#1f77b4')
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("d_eff(t)", fontsize=12)
        ax.set_title(f"{run_id} - Effective Dimensionality Evolution", fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # For single epoch data, adjust x-axis
        if len(epochs) == 1:
            ax.set_xlim([0.5, 1.5])

        # Use scientific notation if values are large
        if history["d_eff"].max() > 10000:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        plt.tight_layout()

        d_eff_path = run_dir / f"{run_id}_dimensionality_trajectory.pdf"
        plt.savefig(d_eff_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved dimensionality trajectory: {d_eff_path}")
        print(f"  - {d_eff_path}")

    # Complexity threshold trajectory (D-RAdam only)
    if "rho_threshold" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, history["rho_threshold"], label="Complexity Threshold rho_threshold(t)", marker="o", markersize=5, linewidth=2, color="orange")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("rho_threshold(t)", fontsize=12)
        ax.set_title(f"{run_id} - Complexity Threshold Evolution", fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # For single epoch data, adjust x-axis
        if len(epochs) == 1:
            ax.set_xlim([0.5, 1.5])

        plt.tight_layout()

        threshold_path = run_dir / f"{run_id}_complexity_threshold_trajectory.pdf"
        plt.savefig(threshold_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved complexity threshold trajectory: {threshold_path}")
        print(f"  - {threshold_path}")

    # Variance rectification regime transition
    if "variance_rectification_regime_transition" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, history["variance_rectification_regime_transition"], label="Rectification Active Fraction", marker="o", markersize=5, linewidth=2, color="green")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Fraction of Steps in Rectified Regime", fontsize=12)
        ax.set_ylim([0, 1.05])
        ax.set_title(f"{run_id} - Variance Rectification Regime Transition", fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # For single epoch data, adjust x-axis
        if len(epochs) == 1:
            ax.set_xlim([0.5, 1.5])

        plt.tight_layout()

        rectification_path = run_dir / f"{run_id}_rectification_regime_transition.pdf"
        plt.savefig(rectification_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved rectification transition: {rectification_path}")
        print(f"  - {rectification_path}")


def compute_convergence_speed_from_history(history: pd.DataFrame) -> float:
    """Compute convergence speed at epoch 20 from history."""
    if "train_loss" not in history.columns or len(history) < 1:
        return 0.0
    
    loss_at_epoch_1 = history.iloc[0]["train_loss"]
    
    if len(history) > 19:
        loss_at_epoch_20 = history.iloc[19]["train_loss"]
    else:
        loss_at_epoch_20 = history.iloc[-1]["train_loss"]
    
    if loss_at_epoch_1 > 0:
        convergence_speed = max(0.0, 1.0 - loss_at_epoch_20 / loss_at_epoch_1)
    else:
        convergence_speed = 0.0
    
    return convergence_speed


def aggregate_metrics_across_runs(
    all_run_data: List[Dict],
    results_dir: Path,
) -> Dict:
    """Aggregate metrics across all runs for comparison."""
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated = {
        "primary_metric": "convergence_speed_at_epoch_20",
        "metrics": {},
        "best_proposed": None,
        "best_baseline": None,
        "gap": None,
    }
    
    # Collect all metrics by key
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        summary = run_data["summary"]
        history = run_data["history"]
        
        # Compute convergence_speed if not in summary
        if "convergence_speed_at_epoch_20" not in summary:
            convergence_speed_20 = compute_convergence_speed_from_history(history)
            summary["convergence_speed_at_epoch_20"] = convergence_speed_20
        
        for key, value in summary.items():
            if key not in aggregated["metrics"]:
                aggregated["metrics"][key] = {}
            
            aggregated["metrics"][key][run_id] = value
    
    # Find best proposed and baseline runs
    primary_metric = aggregated["primary_metric"]
    
    best_proposed_val = -np.inf
    best_proposed_run = None
    
    best_baseline_val = -np.inf
    best_baseline_run = None
    
    if primary_metric in aggregated["metrics"]:
        metric_values = aggregated["metrics"][primary_metric]
        
        for run_id, value in metric_values.items():
            if "proposed" in run_id:
                if value > best_proposed_val:
                    best_proposed_val = value
                    best_proposed_run = run_id
            elif "comparative" in run_id or "baseline" in run_id:
                if value > best_baseline_val:
                    best_baseline_val = value
                    best_baseline_run = run_id
    
    if best_proposed_run is not None:
        aggregated["best_proposed"] = {
            "run_id": best_proposed_run,
            "value": float(best_proposed_val),
        }
    
    if best_baseline_run is not None:
        aggregated["best_baseline"] = {
            "run_id": best_baseline_run,
            "value": float(best_baseline_val),
        }
    
    # Calculate gap - METRIC DIRECTION: convergence_speed is higher-is-better
    # So positive gap means proposed is better
    if best_proposed_run is not None and best_baseline_run is not None:
        if best_baseline_val > 0:
            gap = (best_proposed_val - best_baseline_val) / best_baseline_val * 100
            aggregated["gap"] = float(gap)
    
    return aggregated


def generate_comparison_figures(
    all_run_data: List[Dict],
    aggregated_metrics: Dict,
    results_dir: Path,
) -> None:
    """Generate comparison figures across all runs."""
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
    })

    run_ids = [run_data["run_id"] for run_data in all_run_data]
    convergence_speeds_20 = []
    final_accuracies = []
    wall_clock_times = []

    for run_data in all_run_data:
        summary = run_data["summary"]
        convergence_speeds_20.append(
            summary.get("convergence_speed_at_epoch_20", 0.0)
        )
        final_accuracies.append(
            summary.get("final_test_accuracy", 0.0)
        )
        wall_clock_times.append(
            summary.get("wall_clock_training_time", 0.0)
        )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.8, label="Proposed (D-RAdam)"),
        Patch(facecolor="#1f77b4", alpha=0.8, label="Baseline/Comparative"),
    ]

    # Convergence speed comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2ca02c" if "proposed" in rid else "#1f77b4" for rid in run_ids]
    bars = ax.bar(range(len(run_ids)), convergence_speeds_20, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    for i, (bar, value) in enumerate(zip(bars, convergence_speeds_20)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{value:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight='bold')

    ax.set_xlabel("Run ID", fontsize=12, fontweight='bold')
    ax.set_ylabel("Convergence Speed @ Epoch 20", fontsize=12, fontweight='bold')
    ax.set_title("Convergence Speed Comparison Across Runs", fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    comparison_convergence_path = comparison_dir / "comparison_convergence_speed_bar.pdf"
    plt.savefig(comparison_convergence_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved convergence comparison: {comparison_convergence_path}")
    print(f"  - {comparison_convergence_path}")

    # Final accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2ca02c" if "proposed" in rid else "#1f77b4" for rid in run_ids]
    bars = ax.bar(range(len(run_ids)), final_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    for i, (bar, value) in enumerate(zip(bars, final_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{value:.2f}%",
                ha="center", va="bottom", fontsize=11, fontweight='bold')

    ax.set_xlabel("Run ID", fontsize=12, fontweight='bold')
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title("Final Accuracy Comparison Across Runs", fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax.legend(handles=legend_elements, loc="best", framealpha=0.9)

    plt.tight_layout()
    comparison_accuracy_path = comparison_dir / "comparison_final_accuracy_bar.pdf"
    plt.savefig(comparison_accuracy_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved accuracy comparison: {comparison_accuracy_path}")
    print(f"  - {comparison_accuracy_path}")

    # Wall-clock time comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2ca02c" if "proposed" in rid else "#1f77b4" for rid in run_ids]
    bars = ax.bar(range(len(run_ids)), wall_clock_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    for i, (bar, value) in enumerate(zip(bars, wall_clock_times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{value:.2f}s",
                ha="center", va="bottom", fontsize=11, fontweight='bold')

    ax.set_xlabel("Run ID", fontsize=12, fontweight='bold')
    ax.set_ylabel("Wall-Clock Training Time (seconds)", fontsize=12, fontweight='bold')
    ax.set_title("Training Time Comparison Across Runs", fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    comparison_time_path = comparison_dir / "comparison_wall_clock_time_bar.pdf"
    plt.savefig(comparison_time_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved time comparison: {comparison_time_path}")
    print(f"  - {comparison_time_path}")


def main():
    """Main evaluation script."""
    args = parse_arguments()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse run IDs from JSON string
    run_ids = json.loads(args.run_ids)
    logger.info(f"Evaluating runs: {run_ids}")
    
    # Load WandB configuration
    wandb_config = load_wandb_config()
    entity = wandb_config.get("entity", "gengaru617-personal")
    project = wandb_config.get("project", "2025-11-19")
    
    logger.info(f"WandB entity: {entity}, project: {project}")
    
    # Retrieve data for all runs
    all_run_data = []
    for run_id in run_ids:
        run_data = retrieve_run_data(entity, project, run_id, results_dir)
        if run_data is not None:
            all_run_data.append(run_data)
        else:
            logger.warning(f"Failed to retrieve data for run: {run_id}")
    
    if not all_run_data:
        logger.error("No runs retrieved successfully")
        sys.exit(1)
    
    logger.info(f"Retrieved data for {len(all_run_data)} runs")
    
    # Per-run processing
    logger.info("Processing per-run metrics and figures...")
    print("\nGenerated per-run files:")
    for run_data in all_run_data:
        export_per_run_metrics(run_data, results_dir)
        generate_per_run_figures(run_data, results_dir)
    
    # Aggregated analysis
    logger.info("Generating aggregated analysis...")
    aggregated_metrics = aggregate_metrics_across_runs(all_run_data, results_dir)
    
    # Save aggregated metrics
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated_metrics_path = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_metrics_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
    logger.info(f"Saved aggregated metrics: {aggregated_metrics_path}")
    print(f"\nGenerated comparison files:")
    print(f"  - {aggregated_metrics_path}")
    
    # Generate comparison figures
    logger.info("Generating comparison figures...")
    generate_comparison_figures(all_run_data, aggregated_metrics, results_dir)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Primary Metric: {aggregated_metrics['primary_metric']}")
    if aggregated_metrics["best_proposed"]:
        logger.info(f"Best Proposed: {aggregated_metrics['best_proposed']['run_id']} "
                   f"({aggregated_metrics['best_proposed']['value']:.4f})")
    if aggregated_metrics["best_baseline"]:
        logger.info(f"Best Baseline: {aggregated_metrics['best_baseline']['run_id']} "
                   f"({aggregated_metrics['best_baseline']['value']:.4f})")
    if aggregated_metrics["gap"] is not None:
        logger.info(f"Performance Gap: {aggregated_metrics['gap']:.2f}%")
    logger.info("="*80)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
