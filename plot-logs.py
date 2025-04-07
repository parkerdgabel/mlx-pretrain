import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from pathlib import Path

def process_log(log_file: Path) -> tuple[list, list, list, list, list]:
    """Process a single log file and return tokens, training losses, and validation data."""
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Parse training losses from regular log entries
    train_losses = []
    tokens = [0]
    
    # Parse validation losses
    val_steps = []
    val_losses = []
    
    for line in lines:
        if line.startswith("Step") and "validation:" not in line:
            # Regular training log
            parts = line.split("|")
            # First part contains loss
            loss_part = next((p for p in parts if "loss=" in p), None)
            if loss_part:
                loss = float(loss_part.split("=")[1].strip())
                train_losses.append(loss)
                
                # Find tokens processed
                toks_part = next((p for p in parts if "toks=" in p), None)
                if toks_part:
                    toks = float(toks_part.split("=")[1].strip())
                    tokens.append(toks + tokens[-1])
        
        elif "validation:" in line:
            # Validation log
            step = int(line.split()[1])
            val_loss = float(line.split("val_loss=")[1].split()[0])
            val_steps.append(step)
            val_losses.append(val_loss)
    
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
    ema = 0.99
    smoothed_train_losses = [train_losses[0]]
    for loss in train_losses[1:]:
        smoothed_train_losses.append(ema * smoothed_train_losses[-1] + (1 - ema) * loss)
    
    # EMA smoothing for validation loss
    ema_val = 0.9
    smoothed_val_losses = []
    if val_losses:
        smoothed_val_losses = [val_losses[0]]
        for loss in val_losses[1:]:
            smoothed_val_losses.append(ema_val * smoothed_val_losses[-1] + (1 - ema_val) * loss)
    
    return tokens, smoothed_train_losses, val_steps, val_losses, smoothed_val_losses

def main():
    parser = argparse.ArgumentParser(description='Plot training logs for multiple runs')
    parser.add_argument('run_names', type=str, nargs='+', help='Names of the training runs to plot')
    args = parser.parse_args()

    # Create a figure with 2 rows, 2 columns
    plt.figure(figsize=(16, 12))
    
    # Full range training loss plot
    plt.subplot(2, 2, 1)
    for run_name in args.run_names:
        log_file = Path("runs") / run_name / "log.txt"
        if not log_file.exists():
            print(f"Error: Log file not found at {log_file}")
            continue
            
        tokens, train_losses, val_steps, val_losses, _ = process_log(log_file)
        plt.plot(tokens, train_losses, label=f"{run_name} (train)")
    
    plt.xlabel("Total tokens processed")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Total tokens processed")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Last 80% training loss plot
    plt.subplot(2, 2, 2)
    for run_name in args.run_names:
        log_file = Path("runs") / run_name / "log.txt"
        if not log_file.exists():
            continue
            
        tokens, train_losses, _, _, _ = process_log(log_file)
        cutoff = int(0.2*len(tokens))
        plt.plot(tokens[cutoff:], train_losses[cutoff:], label=f"{run_name} (train)")
    
    plt.xlabel("Total tokens processed")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Total tokens processed (last 80%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation loss
    plt.subplot(2, 2, 3)
    has_validation_data = False
    for run_name in args.run_names:
        log_file = Path("runs") / run_name / "log.txt"
        if not log_file.exists():
            continue
            
        tokens, _, val_steps, val_losses, smoothed_val_losses = process_log(log_file)
        if val_steps and val_losses:
            has_validation_data = True
            # Map validation steps to tokens for x-axis
            step_to_token = {}
            for i, step in enumerate(tokens):
                step_to_token[i] = step
                
            # Estimate token count for validation steps
            val_tokens = []
            for step in val_steps:
                if step < len(tokens):
                    val_tokens.append(tokens[step])
                else:
                    # Estimate based on last available token count
                    val_tokens.append(tokens[-1] * step / len(tokens))
            
            plt.plot(val_tokens, val_losses, 'o', alpha=0.5, label=f"{run_name} (val)")
            plt.plot(val_tokens, smoothed_val_losses, '-', label=f"{run_name} (val EMA)")
    
    if has_validation_data:
        plt.xlabel("Total tokens (estimated)")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.title("No validation data available")
    
    # Plot training vs validation loss for the best run
    plt.subplot(2, 2, 4)
    best_run = None
    best_final_val_loss = float('inf')
    
    for run_name in args.run_names:
        log_file = Path("runs") / run_name / "log.txt"
        if not log_file.exists():
            continue
            
        _, _, val_steps, val_losses, _ = process_log(log_file)
        if val_steps and val_losses and val_losses[-1] < best_final_val_loss:
            best_final_val_loss = val_losses[-1]
            best_run = run_name
    
    if best_run:
        log_file = Path("runs") / best_run / "log.txt"
        tokens, train_losses, val_steps, val_losses, smoothed_val_losses = process_log(log_file)
        
        # Plot training curve
        plt.plot(tokens, train_losses, label=f"{best_run} (train)")
        
        # Plot validation points
        val_tokens = []
        for step in val_steps:
            if step < len(tokens):
                val_tokens.append(tokens[step])
            else:
                # Estimate based on last available token count
                val_tokens.append(tokens[-1] * step / len(tokens))
        
        plt.plot(val_tokens, val_losses, 'ro', alpha=0.5, label=f"{best_run} (val)")
        plt.plot(val_tokens, smoothed_val_losses, 'r-', label=f"{best_run} (val EMA)")
        plt.xlabel("Total tokens processed")
        plt.ylabel("Loss")
        plt.title(f"Training vs Validation Loss for Best Run: {best_run}")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.title("No validation data available for comparison")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
