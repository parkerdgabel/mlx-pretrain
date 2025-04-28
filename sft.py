#!/usr/bin/env python
"""
Script for supervised fine-tuning (SFT) of models pretrained with mlx-pretrain.

Usage:
    python sft.py --config <config_path> [--pretrained_model <model_path>]

Example:
    python sft.py --config sft-config.yaml --pretrained_model "runs/Llama (2M)"
"""

import json
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import yaml
import mlx.optimizers as optim
import mlx_optimizers as optim_x
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import os
import importlib
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import inspect

# Import from existing project
from train import (
    Config, TokenizerManager, CheckpointManager, 
    filter_valid_args, OptimizationManager,
    ModelConfig, TrainingConfig, LoggingConfig, 
    SystemConfig, ResumeConfig
)


@dataclass
class SFTDataConfig:
    input_file: str
    validation_file: Optional[str] = None
    tokenizer_path: Optional[str] = None
    preprocessing: Dict[str, Any] = None
    tokenizer: Dict[str, Any] = None
    weight_path: Optional[str] = None

    # SFT specific fields
    prompt_format: str = "{instruction}"
    response_format: str = "{response}"
    system_prompt: Optional[str] = None
    input_field: str = "instruction"
    output_field: str = "response"
    input_field_optional: bool = False


@dataclass
class SFTConfig(Config):
    data: SFTDataConfig

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert nested dictionaries to appropriate dataclasses
        data_config = SFTDataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        resume_config = None
        if 'resume' in config_dict:
            resume_config = ResumeConfig(**config_dict.get('resume', {}))

        return cls(
            name=config_dict.get('name', 'Unnamed SFT Run'),
            overwrite=config_dict.get('overwrite', False),
            data=data_config,
            model=model_config,
            training=training_config,
            logging=logging_config,
            system=system_config,
            resume=resume_config
        )


class SFTDataManager:
    """Manages data loading and batch generation for supervised fine-tuning."""

    def __init__(self, config: SFTDataConfig, tokenizer: TokenizerManager, batch_size: int = 1):
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.train_data = []
        self.val_data = []
        self.val_ptr = 0

        # Load preprocessing config
        self.max_context_size = config.preprocessing.get('max_context_size', 1024)
        self.chunk_overlap = config.preprocessing.get('chunk_overlap', 0)

        # Format templates
        self.prompt_format = config.prompt_format
        self.response_format = config.response_format
        self.system_prompt = config.system_prompt

        # Field names
        self.input_field = config.input_field
        self.output_field = config.output_field
        self.input_field_optional = config.input_field_optional

    def load_data(self):
        """Load training and validation data."""
        print(f"Loading training data from {self.config.input_file}")
        self._load_file(self.config.input_file, self.train_data)

        if self.config.validation_file:
            print(f"Loading validation data from {self.config.validation_file}")
            self._load_file(self.config.validation_file, self.val_data)

        print(f"Loaded {len(self.train_data)} training examples and {len(self.val_data)} validation examples")
        return len(self.train_data), len(self.val_data)

    def _load_file(self, file_path: str, data_list: list):
        """Load data from a JSONL file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())

                    # Check if required fields exist
                    if self.output_field not in item:
                        continue

                    if not self.input_field_optional and self.input_field not in item:
                        continue

                    # Format the prompt and response
                    instruction = item.get(self.input_field, "")
                    response = item[self.output_field]

                    # Format according to templates
                    prompt = self.prompt_format.format(instruction=instruction)
                    if self.system_prompt:
                        prompt = f"{self.system_prompt}\n{prompt}"

                    formatted_response = self.response_format.format(response=response)

                    # Tokenize and add to data list
                    prompt_tokens = self.tokenizer.tokenize(prompt)
                    response_tokens = self.tokenizer.tokenize(formatted_response)

                    # Check if the combined length is within max context size
                    if len(prompt_tokens) + len(response_tokens) <= self.max_context_size:
                        data_list.append({
                            "prompt": prompt_tokens,
                            "response": response_tokens,
                            "combined": prompt_tokens + response_tokens
                        })
                except Exception as e:
                    print(f"Error processing line: {e}")

    def generate_batch(self, step: int) -> mx.array:
        """Generate a batch of training data."""
        # Randomly sample examples
        indices = random.sample(range(len(self.train_data)), min(self.batch_size, len(self.train_data)))
        examples = [self.train_data[i] for i in indices]
        return self._create_batch(examples)

    def generate_validation_batch(self, batch_idx: int) -> mx.array:
        """Generate a batch of validation data."""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.val_data))
        examples = self.val_data[start_idx:end_idx]
        return self._create_batch(examples)

    def _create_batch(self, examples: list) -> mx.array:
        """Create a batch from a list of examples."""
        # Pad sequences to the same length
        max_len = max(len(ex["combined"]) for ex in examples)
        batch = []

        for ex in examples:
            # Pad with PAD_TOKEN
            padded = ex["combined"] + [self.tokenizer.PAD_TOKEN] * (max_len - len(ex["combined"]))
            batch.append(padded)

        return mx.array(batch)

    def has_validation_data(self) -> bool:
        """Check if validation data is available."""
        return len(self.val_data) > 0

    def num_validation_batches(self) -> int:
        """Get the number of validation batches."""
        return (len(self.val_data) + self.batch_size - 1) // self.batch_size


class SFTTrainer:
    """Trainer for supervised fine-tuning."""

    def __init__(self, config_path: str, pretrained_model_path: Optional[str] = None):
        self.config_path = config_path
        self.pretrained_model_path = pretrained_model_path
        self.config = SFTConfig.from_yaml(config_path)

        # Set up run directory
        self.run_dir = CheckpointManager.setup_run_directory(self.config.name)
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.run_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / 'training.log'

        # Initialize components
        self.setup_system()
        self.tokenizer = TokenizerManager(self.config.data, self.run_dir)
        self.data_manager = SFTDataManager(
            self.config.data, 
            self.tokenizer, 
            self.config.training.hyperparameters['batch_size']
        )

        # Load data
        self.data_manager.load_data()

        # Set up model
        self.setup_model()

        # Set up training
        self.setup_training()

        # Set up logging
        self.setup_logging()

        # Initialize tracking variables
        self.total_tokens = mx.array(0, dtype=mx.int32)
        self.validation_losses = []
        self.start_step = 0

    def setup_system(self):
        # Set random seeds
        random.seed(self.config.system.seed)
        np.random.seed(self.config.system.seed)
        mx.random.seed(self.config.system.seed)

    def setup_model(self):
        model_cfg = self.config.model
        arch_file = f"arch.{model_cfg.architecture}"
        mlx_lm_file = f"mlx_lm.models.{model_cfg.architecture}"
        Model = None
        ModelArgs = None

        try:
            module = importlib.import_module(arch_file)
            Model = getattr(module, 'Model')
            ModelArgs = getattr(module, 'ModelArgs')
        except ImportError:
            try:
                module = importlib.import_module(mlx_lm_file)
                Model = getattr(module, 'Model')
                ModelArgs = getattr(module, 'ModelArgs')
            except ImportError:
                raise ImportError(f"Model architecture '{model_cfg.architecture}' not found in both {arch_file} and {mlx_lm_file}")

        all_args = {
            'model_type': model_cfg.architecture,
            'hidden_size': model_cfg.dimensions['hidden_size'],
            'num_hidden_layers': model_cfg.dimensions.get('num_layers', 8),
            'intermediate_size': model_cfg.dimensions['intermediate_size'],
            'num_attention_heads': model_cfg.attention['num_heads'],
            'rms_norm_eps': model_cfg.normalization['rms_norm_eps'],
            'vocab_size': self.tokenizer.VOCAB_SIZE,
            'head_dim': model_cfg.attention['head_dim'],
            'max_position_embeddings': model_cfg.attention['max_position_embeddings'],
            'num_key_value_heads': model_cfg.attention['num_kv_heads'],
            'attention_bias': model_cfg.misc['attention_bias'],
            'mlp_bias': model_cfg.misc['mlp_bias'],
            'rope_theta': model_cfg.rope['theta'],
            'rope_traditional': model_cfg.rope['traditional'],
            'rope_scaling': model_cfg.rope['scaling'],
            'tie_word_embeddings': model_cfg.misc['tie_word_embeddings'],
            'logit_scale': model_cfg.misc.get('logit_scale', None),
            'num_local_experts': model_cfg.misc.get('num_local_experts', 0),
            'num_experts_per_tok': model_cfg.misc.get('num_experts_per_tok', 0),
        }
        valid_args = filter_valid_args(ModelArgs, all_args)
        args = ModelArgs(**valid_args)

        self.model = Model(args)

        # Load pretrained weights if specified
        if self.pretrained_model_path:
            pretrained_path = Path(self.pretrained_model_path)

            # Check if it's a run directory or a direct checkpoint path
            if pretrained_path.is_dir():
                # Find the latest checkpoint
                metadata_path = pretrained_path / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    if 'checkpoints' in metadata and metadata['checkpoints']:
                        # Get the latest checkpoint
                        latest_checkpoint = metadata['checkpoints'][-1]
                        model_path = pretrained_path / latest_checkpoint['paths']['model']
                        print(f"Loading pretrained weights from {model_path}")
                        weights = mx.load(str(model_path))
                        self.model.update(weights)
                else:
                    raise ValueError(f"No metadata.json found in {pretrained_path}")
            else:
                # Direct checkpoint path
                print(f"Loading pretrained weights from {pretrained_path}")
                weights = mx.load(str(pretrained_path))
                self.model.update(weights)

        # Log model size
        p = sum(v.size for _, v in tree_flatten(self.model.trainable_parameters())) / 10**6
        print(f"Model has {p:.2f}M parameters")

    def setup_training(self):
        # Calculate number of training steps
        num_samples = len(self.data_manager.train_data)
        batch_size = self.config.training.hyperparameters['batch_size']
        steps_per_epoch = num_samples // batch_size

        if self.config.training.epochs is not None:
            # If epochs is set, calculate total steps based on epochs
            self.total_steps = steps_per_epoch * self.config.training.epochs
        else:
            # Otherwise use specified iters or default to one epoch
            self.total_steps = self.config.training.hyperparameters.get('iters', steps_per_epoch)

        # Store steps_per_epoch for logging
        self.steps_per_epoch = steps_per_epoch

        # Setup optimization
        opt_manager = OptimizationManager(self.config.training, self.total_steps)
        self.lr_schedule = opt_manager.create_scheduler()
        self.optimizer = opt_manager.create_optimizer(self.lr_schedule)

    def setup_logging(self):
        # Create initial metadata file
        metadata = {
            'name': self.config.name,
            'created_at': datetime.now().isoformat(),
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__,
                'system': self.config.system.__dict__
            },
            'training_info': {
                'steps_per_epoch': self.steps_per_epoch,
                'total_steps': self.total_steps,
                'epochs': self.config.training.epochs
            },
            'sft_specific': {
                'prompt_format': self.config.data.prompt_format,
                'response_format': self.config.data.response_format,
                'system_prompt': self.config.data.system_prompt,
            }
        }

        # Add tokenizer information to metadata
        if self.config.data.tokenizer_path:
            metadata['tokenizer'] = {
                'type': 'external',
                'path': self.config.data.tokenizer_path,
                'vocab_size': self.tokenizer.VOCAB_SIZE
            }
        else:
            metadata['tokenizer'] = {
                'type': 'byte-level',
                'vocab_size': self.tokenizer.VOCAB_SIZE
            }

        # Add pretrained model info if applicable
        if self.pretrained_model_path:
            metadata['pretrained_model'] = str(self.pretrained_model_path)

        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save the config used to the run directory
        with open(self.run_dir / 'config.yaml', 'w') as f:
            with open(self.config_path, 'r') as config_file:
                f.write(config_file.read())

    def compute_loss(self, model, inputs: mx.array, targets: mx.array) -> Tuple[mx.array, int]:
        logits = model(inputs)
        logits = logits.astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, targets)

        # Mask padding tokens
        pad_mask = (targets != self.tokenizer.PAD_TOKEN)
        loss = loss * pad_mask
        ntoks = pad_mask.sum()

        return loss.sum() / ntoks, ntoks

    def validate(self) -> float:
        """Run validation on the validation dataset."""
        if not self.data_manager.has_validation_data():
            return None

        # Ensure we're in evaluation mode (no need for gradients)
        total_loss = 0.0
        total_tokens = 0

        # Process all validation batches
        num_batches = min(self.data_manager.num_validation_batches(), 50)  # Cap at 50 batches

        for batch_idx in range(num_batches):
            batch = self.data_manager.generate_validation_batch(batch_idx)

            # Forward pass only
            loss, tokens = self.compute_loss(self.model, batch[:, :-1], batch[:, 1:])

            # Accumulate metrics
            total_loss += float(loss)
            total_tokens += tokens

            # Clear GPU cache if needed
            if self.config.system.device == "gpu":
                mx.clear_cache()

        # Calculate average loss
        avg_loss = total_loss / num_batches

        return avg_loss

    def save_checkpoint(self, step: int | str, val_loss: float = None):
        # Save model weights
        weights = dict(tree_flatten(self.model.parameters()))
        model_path = self.checkpoint_dir / f'step_{step}_model.safetensors'
        mx.save_safetensors(str(model_path), weights)

        # Save optimizer state
        optimizer_state = dict(tree_flatten(self.optimizer.state))
        optimizer_path = self.checkpoint_dir / f'step_{step}_optimizer.safetensors'
        mx.save_safetensors(str(optimizer_path), optimizer_state)

        # Save training state
        training_state = {
            'step': step if isinstance(step, int) else self.total_steps,
            'val_ptr': self.data_manager.val_ptr,
            'total_tokens': self.total_tokens.item(),
            'validation_losses': self.validation_losses,
        }
        state_path = self.checkpoint_dir / f'step_{step}_state.json'
        with open(state_path, 'w') as f:
            json.dump(training_state, f)

        # Update metadata with checkpoint info
        metadata_path = self.run_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if 'checkpoints' not in metadata:
            metadata['checkpoints'] = []

        checkpoint_info = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'paths': {
                'model': f'checkpoints/step_{step}_model.safetensors',
                'optimizer': f'checkpoints/step_{step}_optimizer.safetensors',
                'state': f'checkpoints/step_{step}_state.json'
            }
        }

        # Include validation loss if available
        if val_loss is not None:
            checkpoint_info['validation_loss'] = val_loss

        metadata['checkpoints'].append(checkpoint_info)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def log_metrics(self, step: int, loss: float, tokens: int, 
                   total_tokens: int, start_time: float, val_loss: float = None) -> str:
        metrics = []

        # Add epoch information if epochs are configured
        if self.config.training.epochs is not None:
            current_epoch = step // self.steps_per_epoch + 1
            epoch_step = step % self.steps_per_epoch + 1
            metrics.append(f"epoch={current_epoch}/{self.config.training.epochs} ({epoch_step}/{self.steps_per_epoch})")

        if self.config.logging.metrics['log_loss']:
            metrics.append(f"loss={loss:.3e}")

            # Add validation loss if available
            if val_loss is not None:
                metrics.append(f"val_loss={val_loss:.3e}")

        if self.config.logging.metrics['log_perplexity']:
            metrics.append(f"ppl={np.exp(loss):.2f}")
            if val_loss is not None:
                metrics.append(f"val_ppl={np.exp(val_loss):.2f}")

        if self.config.logging.metrics['log_tokens_per_second']:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            metrics.append(f"tok/s={tokens_per_sec:.1f}")

        if self.config.logging.metrics['log_learning_rate']:
            lr = self.lr_schedule(step)
            metrics.append(f"lr={lr:.2e}")

        if self.config.logging.metrics['log_tokens_processed']:
            metrics.append(f"tokens={total_tokens}")

        return " | ".join(metrics)

    def train(self):
        # Initialize variables
        total_tokens = self.total_tokens
        start_step = 0

        # Check if resuming from checkpoint
        if self.config.resume and self.config.resume.checkpoint:
            checkpoint_path = self.config.resume.checkpoint
            reset_optimizer = self.config.resume.reset_optimizer
            # Implement checkpoint loading if needed

        loss_value_and_grad = nn.value_and_grad(self.model, self.compute_loss)
        start_time = time.time()

        # Create progress bar
        progress_bar = tqdm(range(self.total_steps), desc="Fine-tuning", initial=start_step)

        # Initialize logging
        with open(self.log_file, 'a' if start_step > 0 else 'w') as log_file:
            if start_step == 0:
                log_file.write(f"SFT started at {datetime.now()}\n")
                log_file.write(f"Total steps: {self.total_steps}\n")
                if self.config.training.epochs is not None:
                    log_file.write(f"Training for {self.config.training.epochs} epochs with {self.steps_per_epoch} steps per epoch\n")
                if self.data_manager.has_validation_data():
                    log_file.write(f"Validation data: {self.config.data.validation_file}\n")
                    log_file.write(f"Validation batches: {self.data_manager.num_validation_batches()}\n")
                log_file.write("=" * 50 + "\n\n")

            # Log initial validation loss if validation data is available
            val_loss = None
            if self.config.logging.steps.get('validation_interval', 0) > 0 and self.data_manager.has_validation_data():
                val_loss = self.validate()
                log_file.write(f"Initial validation loss: {val_loss:.4e} (ppl={np.exp(val_loss):.2f})\n\n")
                # Add to validation loss history
                self.validation_losses.append((0, val_loss))

            for step in progress_bar:
                step += start_step
                if step >= self.total_steps:
                    break

                # Generate batch
                batch = self.data_manager.generate_batch(step)

                # Forward and backward pass
                (loss, tokens), grad = loss_value_and_grad(
                    self.model, batch[:, :-1], batch[:, 1:]
                )

                # Gradient clipping if configured
                if 'gradient_clip' in self.config.training.hyperparameters:
                    clip_value = self.config.training.hyperparameters['gradient_clip']
                    grad = tree_map(lambda x: mx.clip(x, -clip_value, clip_value), grad)

                # Update model
                total_tokens += tokens
                self.optimizer.update(self.model, grad)
                mx.eval(loss)

                if self.config.system.device == "gpu":
                    mx.clear_cache()

                # Run validation
                validation_interval = self.config.logging.steps.get('validation_interval', 0)
                if validation_interval > 0 and self.data_manager.has_validation_data() and (step + 1) % validation_interval == 0:
                    val_loss = self.validate()
                    # Add to validation loss history
                    self.validation_losses.append((step + 1, val_loss))

                    # Log validation separately for clear visibility
                    val_metrics = f"val_loss={val_loss:.3e} | val_ppl={np.exp(val_loss):.2f}"
                    log_file.write(f"Step {step + 1} validation: {val_metrics}\n")
                    log_file.flush()

                # Logging
                if step % self.config.logging.steps['logging_interval'] == 0:
                    # Only include val_loss if it was just calculated
                    current_val_loss = val_loss if validation_interval > 0 and (step + 1) % validation_interval == 0 else None
                    metrics = self.log_metrics(step, loss, tokens, total_tokens, start_time, current_val_loss)

                    # Update progress bar
                    progress_bar.set_description(metrics)

                    # Write to log file
                    log_message = f"Step {step}: {metrics}\n"
                    log_file.write(log_message)
                    log_file.flush()

                # Save checkpoint
                if (1 + step) % self.config.logging.steps['checkpoint_interval'] == 0:
                    # Find the most recent validation loss if available
                    last_val_loss = val_loss if val_loss is not None else None
                    # Update total_tokens in the trainer instance for checkpoint saving
                    self.total_tokens = total_tokens
                    self.save_checkpoint(step + 1, last_val_loss)

        # Final validation
        final_val_loss = None
        if validation_interval > 0 and self.data_manager.has_validation_data():
            final_val_loss = self.validate()
            self.validation_losses.append((self.total_steps, final_val_loss))

        # Save final checkpoint
        self.total_tokens = total_tokens
        self.save_checkpoint("final", final_val_loss)

        # Final log message
        with open(self.log_file, 'a') as log_file:
            log_file.write("\n" + "=" * 50 + "\n")
            log_file.write(f"Training completed at {datetime.now()}\n")
            log_file.write(f"Total tokens processed: {total_tokens}\n")
            if final_val_loss is not None:
                log_file.write(f"Final validation loss: {final_val_loss:.4e} (ppl={np.exp(final_val_loss):.2f})\n")
            log_file.write("=" * 50 + "\n")

        print(f"\nTraining completed. Model saved to {self.run_dir}")
        if final_val_loss is not None:
            print(f"Final validation loss: {final_val_loss:.4e} (ppl={np.exp(final_val_loss):.2f})")

        return self.run_dir


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for MLX models")
    parser.add_argument("--config", required=True, help="Path to SFT config YAML file")
    parser.add_argument("--pretrained_model", help="Path to pretrained model directory or checkpoint")
    args = parser.parse_args()

    trainer = SFTTrainer(args.config, args.pretrained_model)
    trainer.train()


if __name__ == "__main__":
    main()
