#!/usr/bin/env python
"""
Plot training logs for MLX-Pretrain models.

This script visualizes training metrics for pretraining, supervised fine-tuning (SFT),
and reinforcement learning (RL) runs. It automatically detects the type of training
and plots appropriate metrics.

Usage:
    python plot-logs.py <run_names> [--no-val] [--metrics {loss,rewards,kl,policy}]

Examples:
    # Plot loss for a pretraining run
    python plot-logs.py "Llama (2M)"

    # Plot loss for multiple runs (can mix pretraining, SFT, and RL)
    python plot-logs.py "Llama (2M)" "Llama-2M-SFT" "Llama-2M-RL"

    # Plot rewards for an RL run
    python plot-logs.py "Llama-2M-RL" --metrics rewards

    # Plot KL divergence for an RL run
    python plot-logs.py "Llama-2M-RL" --metrics kl
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

def detect_training_type(run_dir: Path) -> str:
    """Detect the type of training (pretraining, SFT, or RL) based on available files."""
    # Check for metrics.json (RL)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        if 'rewards' in metrics and len(metrics['rewards']) > 0:
            return "rl"

    # Check config.yaml for training type
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                if 'data' in config and 'prompt_format' in config['data']:
                    return "sft"
            except:
                pass

    # Default to pretraining
    return "pretraining"

def process_rl_metrics(metrics_file: Path) -> Dict[str, Any]:
    """Process RL metrics from metrics.json file."""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Calculate tokens for x-axis
    tokens = []
    total_tokens = metrics.get('total_tokens', 0)
    steps = metrics.get('steps', [])

    if steps:
        # Estimate tokens per step
        tokens_per_step = total_tokens / steps[-1] if steps[-1] > 0 else 0
        tokens = [s * tokens_per_step for s in steps]

    # Apply EMA smoothing to metrics
    ema = 0.9
    smoothed_metrics = {}

    for key in ['train_loss', 'rewards', 'policy_loss', 'kl_divergence']:
        if key in metrics and metrics[key]:
            smoothed_metrics[key] = [metrics[key][0]]
            for value in metrics[key][1:]:
                smoothed_metrics[key].append(ema * smoothed_metrics[key][-1] + (1 - ema) * value)

    # Add tokens and steps to the result
    result = {
        'tokens': tokens,
        'steps': steps,
        **smoothed_metrics
    }

    # Add validation metrics if available
    if 'val_loss' in metrics and metrics['val_loss']:
        result['val_steps'] = [steps[i] for i in range(len(steps)) if i < len(metrics['val_loss'])]
        result['val_loss'] = metrics['val_loss']

        # Apply EMA smoothing to validation loss
        ema_val = 0.0
        result['smoothed_val_loss'] = [metrics['val_loss'][0]]
        for loss in metrics['val_loss'][1:]:
            result['smoothed_val_loss'].append(ema_val * result['smoothed_val_loss'][-1] + (1 - ema_val) * loss)
            ema_val = ema ** (1000/16)

    return result

def process_log(log_file: Path) -> tuple[list, list, list, list, list]:
    """Process a single log file and return tokens, training losses, and validation data."""
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Parse training losses from regular log entries
    train_steps = []

    # Parse validation losses
    val_steps = []
    val_losses = []

    for line in lines:
        if line.startswith("Step") and "validation:" not in line:
            step = int(line.split()[1][:-1])
            # Regular training log
            parts = line.split("|")
            # First part contains loss
            loss_part = next((p for p in parts if "loss=" in p), None)
            if loss_part:
                loss = float(loss_part.split("=")[1].strip())

                # Find tokens processed
                toks_part = next((p for p in parts if "toks=" in p or "tokens=" in p), None)
                if toks_part:
                    toks_key = "toks=" if "toks=" in toks_part else "tokens="
                    toks = float(toks_part.split(toks_key)[1].strip())
                    train_steps.append((step, loss, toks))

        elif "validation:" in line:
            # Validation log
            step = int(line.split()[1])
            val_loss_part = next((p for p in line.split("|") if "val_loss=" in p), None)
            if val_loss_part:
                val_loss = float(val_loss_part.split("=")[1].strip())
                val_steps.append(step)
                val_losses.append(val_loss)

    # Sort train_steps
    train_steps.sort(key=lambda x: x[0])
    deduped_train_steps = []
    for step, loss, toks in train_steps:
        if len(deduped_train_steps) == 0 or deduped_train_steps[-1][0] != step:
            deduped_train_steps.append((step, loss, toks))

    train_losses = []
    tokens = [0]
    for step, loss, toks in deduped_train_steps:
        train_losses.append(loss)
        # Append tokens processed
        tokens.append(toks + tokens[-1])

    # Ensure tokens list has same length as losses
    if len(tokens) > len(train_losses) + 1:
        tokens = tokens[:len(train_losses) + 1]
    tokens = tokens[1:]

    # Validation data might also be in metadata
    metadata_path = log_file.parent / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            if 'validation' in metadata and len(metadata['validation']['steps']) > 0:
                # Use metadata for validation data as it's more reliable
                val_steps = metadata['validation']['steps']
                val_losses = metadata['validation']['losses']
        except:
            # Fallback to log-parsed validation data
            pass

    # EMA smoothing for training loss
    ema = 0.9
    smoothed_train_losses = [train_losses[0]] if train_losses else []
    for loss in train_losses[1:]:
        smoothed_train_losses.append(ema * smoothed_train_losses[-1] + (1 - ema) * loss)

    # EMA smoothing for validation loss
    ema_val = 0.0
    smoothed_val_losses = []
    if val_losses:
        smoothed_val_losses = [val_losses[0]]
        for loss in val_losses[1:]:
            smoothed_val_losses.append(ema_val * smoothed_val_losses[-1] + (1 - ema_val) * loss)
            ema_val = ema ** (1000/16)

    return tokens, smoothed_train_losses, val_steps, val_losses, smoothed_val_losses

def main():
    parser = argparse.ArgumentParser(description='Plot training logs for multiple runs')
    parser.add_argument('run_names', type=str, nargs='+', help='Names of the training runs to plot')
    parser.add_argument('--no-val', action='store_true', help='Ignore validation data when plotting')
    parser.add_argument('--metrics', type=str, choices=['loss', 'rewards', 'kl', 'policy'], default='loss',
                        help='Which metrics to plot (for RL runs)')
    args = parser.parse_args()

    # Determine if we have any RL runs
    has_rl_runs = False
    run_types = {}

    for run_name in args.run_names:
        run_dir = Path("runs") / run_name
        if not run_dir.exists():
            print(f"Error: Run directory not found at {run_dir}")
            continue

        run_type = detect_training_type(run_dir)
        run_types[run_name] = run_type
        if run_type == "rl":
            has_rl_runs = True

    # Create figures based on run types
    if has_rl_runs and args.metrics != 'loss':
        # Create a figure for RL-specific metrics
        plt.figure(figsize=(16, 8))

        # Plot the selected RL metric
        plt.subplot(1, 1, 1)

        for run_name in args.run_names:
            if run_name not in run_types:
                continue

            if run_types[run_name] == "rl":
                metrics_file = Path("runs") / run_name / "metrics.json"
                if not metrics_file.exists():
                    print(f"Error: Metrics file not found at {metrics_file}")
                    continue

                metrics = process_rl_metrics(metrics_file)

                if args.metrics == 'rewards' and 'rewards' in metrics:
                    plt.plot(metrics['tokens'], metrics['rewards'], label=f"{run_name} (rewards)")
                elif args.metrics == 'kl' and 'kl_divergence' in metrics:
                    plt.plot(metrics['tokens'], metrics['kl_divergence'], label=f"{run_name} (KL divergence)")
                elif args.metrics == 'policy' and 'policy_loss' in metrics:
                    plt.plot(metrics['tokens'], metrics['policy_loss'], label=f"{run_name} (policy loss)")

        plt.xlabel("Total tokens processed")
        plt.ylabel(args.metrics.capitalize())
        plt.title(f"{args.metrics.capitalize()} over Training")
        plt.legend()
        plt.grid(True, alpha=0.3)

    else:
        # Create a figure for loss plots
        plt.figure(figsize=(16, 8))

        # Full range training and validation loss plot
        plt.subplot(1, 2, 1)
        has_validation_data = False

        for run_name in args.run_names:
            if run_name not in run_types:
                continue

            run_dir = Path("runs") / run_name
            run_type = run_types[run_name]

            if run_type == "rl":
                # Process RL metrics from metrics.json
                metrics_file = run_dir / "metrics.json"
                if not metrics_file.exists():
                    print(f"Error: Metrics file not found at {metrics_file}")
                    continue

                metrics = process_rl_metrics(metrics_file)

                if 'tokens' in metrics and 'train_loss' in metrics:
                    plt.plot(metrics['tokens'], metrics['train_loss'], label=f"{run_name} (train EMA)")

                    if not args.no_val and 'val_loss' in metrics:
                        has_validation_data = True
                        val_tokens = []
                        for i, step in enumerate(metrics['val_steps']):
                            if i < len(metrics['tokens']):
                                val_tokens.append(metrics['tokens'][i])
                            else:
                                # Estimate based on last available token count
                                val_tokens.append(metrics['tokens'][-1] * step / metrics['steps'][-1])

                        plt.plot(val_tokens, metrics['smoothed_val_loss'], '-', label=f"{run_name} (val EMA)")
            else:
                # Process standard log file
                log_file = run_dir / "log.txt"
                if not log_file.exists():
                    print(f"Error: Log file not found at {log_file}")
                    continue

                tokens, train_losses, val_steps, val_losses, smoothed_val_losses = process_log(log_file)

                if not tokens or not train_losses:
                    print(f"Warning: No training data found for {run_name}")
                    continue

                # Plot training losses
                plt.plot(tokens, train_losses, label=f"{run_name} (train EMA)")

                # Plot validation losses if available and not disabled
                if not args.no_val and val_steps and val_losses:
                    has_validation_data = True
                    val_tokens = []
                    for step in val_steps:
                        if step < len(tokens):
                            val_tokens.append(tokens[step])
                        else:
                            # Estimate based on last available token count
                            val_tokens.append(tokens[-1] * step / len(tokens))

                    plt.plot(val_tokens, smoothed_val_losses, '-', label=f"{run_name} (val EMA)")

        plt.xlabel("Total tokens processed")
        plt.ylabel("Loss")
        title = "Training Loss (Full Range)" if args.no_val else "Training and Validation Loss (Full Range)"
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Last 80% training and validation loss plot
        plt.subplot(1, 2, 2)

        for run_name in args.run_names:
            if run_name not in run_types:
                continue

            run_dir = Path("runs") / run_name
            run_type = run_types[run_name]

            if run_type == "rl":
                # Process RL metrics from metrics.json
                metrics_file = run_dir / "metrics.json"
                if not metrics_file.exists():
                    continue

                metrics = process_rl_metrics(metrics_file)

                if 'tokens' in metrics and 'train_loss' in metrics:
                    # Calculate 20% cutoff point
                    cutoff = int(0.2 * len(metrics['tokens']))
                    tokens_last_80 = metrics['tokens'][cutoff:]
                    train_losses_last_80 = metrics['train_loss'][cutoff:]

                    # Plot training losses for last 80%
                    plt.plot(tokens_last_80, train_losses_last_80, label=f"{run_name} (train EMA)")

                    # Plot validation losses for last 80% if available and not disabled
                    if not args.no_val and 'val_loss' in metrics and 'val_steps' in metrics:
                        val_tokens = []
                        for i, step in enumerate(metrics['val_steps']):
                            if i < len(metrics['tokens']):
                                val_tokens.append(metrics['tokens'][i])
                            else:
                                # Estimate based on last available token count
                                val_tokens.append(metrics['tokens'][-1] * step / metrics['steps'][-1])

                        # Filter validation points to only include those in the last 80%
                        if tokens_last_80:
                            last_80_points = [(t, l, s) for t, l, s in zip(val_tokens, metrics['val_loss'], metrics['smoothed_val_loss']) 
                                            if t >= tokens_last_80[0]]

                            if last_80_points:
                                last_tokens, last_losses, last_smoothed = zip(*last_80_points)
                                plt.plot(last_tokens, last_smoothed, '-', label=f"{run_name} (val EMA)")
            else:
                # Process standard log file
                log_file = run_dir / "log.txt"
                if not log_file.exists():
                    continue

                tokens, train_losses, val_steps, val_losses, smoothed_val_losses = process_log(log_file)

                if not tokens or not train_losses:
                    continue

                # Calculate 20% cutoff point
                cutoff = int(0.2 * len(tokens))
                tokens_last_80 = tokens[cutoff:]
                train_losses_last_80 = train_losses[cutoff:]

                # Plot training losses for last 80%
                plt.plot(tokens_last_80, train_losses_last_80, label=f"{run_name} (train EMA)")

                # Plot validation losses for last 80% if available and not disabled
                if not args.no_val and val_steps and val_losses:
                    val_tokens = []
                    for step in val_steps:
                        if step < len(tokens):
                            val_tokens.append(tokens[step])
                        else:
                            # Estimate based on last available token count
                            val_tokens.append(tokens[-1] * step / len(tokens))

                    # Filter validation points to only include those in the last 80%
                    last_80_points = [(t, l, s) for t, l, s in zip(val_tokens, val_losses, smoothed_val_losses) 
                                    if t >= tokens_last_80[0]]

                    if last_80_points:
                        last_tokens, last_losses, last_smoothed = zip(*last_80_points)
                        plt.plot(last_tokens, last_smoothed, '-', label=f"{run_name} (val EMA)")

        plt.xlabel("Total tokens processed")
        plt.ylabel("Loss")
        title = "Training Loss (Last 80%)" if args.no_val else "Training and Validation Loss (Last 80%)"
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
