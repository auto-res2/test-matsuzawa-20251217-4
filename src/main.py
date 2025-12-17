#!/usr/bin/env python
"""
Main orchestrator for D-RAdam experiment runs.
Loads run configuration from config/runs/{run_id}.yaml and executes training.
Handles mode-based configuration (trial vs full) and calls train.py as subprocess.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging

from omegaconf import DictConfig, OmegaConf
import hydra

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for experiment execution.
    
    Receives run_id, results_dir, and mode via command line.
    Loads run configuration, applies mode-specific adjustments, and spawns training subprocess.
    
    Args:
        cfg: Hydra configuration from config.yaml
    """
    # Validate required CLI parameters
    if not cfg.run.run_id:
        logger.error("run parameter is required. Use: run=<run_id>")
        sys.exit(1)
    
    if not cfg.results_dir:
        logger.error("results_dir parameter is required. Use: results_dir=<path>")
        sys.exit(1)
    
    if not cfg.mode:
        logger.error("mode parameter is required. Use: mode=trial or mode=full")
        sys.exit(1)
    
    run_id = cfg.run.run_id
    results_dir = cfg.results_dir
    mode = cfg.mode
    
    # Validate mode
    if mode not in ["trial", "full"]:
        logger.error(f"Invalid mode: {mode}. Must be 'trial' or 'full'")
        sys.exit(1)
    
    # Load run configuration from config/runs/{run_id}.yaml
    run_config_path = Path("config/runs") / f"{run_id}.yaml"
    if not run_config_path.exists():
        logger.error(f"Run configuration not found: {run_config_path}")
        sys.exit(1)
    
    # Load and merge run config with base config
    run_cfg = OmegaConf.load(run_config_path)
    cfg = OmegaConf.merge(cfg, run_cfg)
    
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Mode-based configuration adjustments
    if mode == "trial":
        cfg.training.epochs = 1
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        logger.info("Trial mode: epochs=1, wandb.mode=disabled, optuna.n_trials=0")
    elif mode == "full":
        cfg.wandb.mode = "online"
        logger.info("Full mode: wandb.mode=online")
    
    logger.info(f"Starting experiment run: {run_id}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Results directory: {results_dir}")
    
    # Prepare configuration to pass to subprocess
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_json = json.dumps(config_dict)
    
    # Execute training script as subprocess with configuration in environment
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXPERIMENT_CONFIG"] = config_json
    env["EXPERIMENT_RUN_ID"] = run_id
    env["EXPERIMENT_MODE"] = mode
    env["EXPERIMENT_RESULTS_DIR"] = results_dir
    
    # Get the absolute path to train.py
    train_script = Path(__file__).parent / "train.py"

    cmd = [
        sys.executable,
        "-u",
        str(train_script),
    ]
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent.parent)
        sys.exit(result.returncode)
    except Exception as e:
        logger.error(f"Error executing training script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
