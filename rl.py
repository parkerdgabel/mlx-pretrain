#!/usr/bin/env python
"""
Script for reinforcement learning fine-tuning (RLHF) of models pretrained with mlx-pretrain.

Usage:
    python rl.py --config <config_path> --pretrained_model <model_path> [--reward_model <reward_model_path>]

Example:
    python rl.py --config rl-config.yaml --pretrained_model "runs/Llama-2M-SFT"
"""

import json
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
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
class RLDataConfig:
    input_file: Optional[str] = None
    validation_file: Optional[str] = None
    tokenizer_path: Optional[str] = None
    preprocessing: Dict[str, Any] = None
    tokenizer: Dict[str, Any] = None
    weight_path: Optional[str] = None

    # Fields for Hugging Face datasets
    hf_dataset_name: Optional[str] = None  # Name of the Hugging Face dataset
    hf_dataset_config: Optional[str] = None  # Configuration name for the dataset
    hf_train_split: Optional[str] = "train"  # Training split name
    hf_val_split: Optional[str] = "validation"  # Validation split name
    use_streaming: bool = False  # Whether to use streaming mode for the dataset

    # RL specific fields
    prompt_format: str = "{instruction}"
    response_format: str = "{response}"
    system_prompt: Optional[str] = None
    input_field: str = "instruction"
    output_field: str = "response"
    reward_field: str = "reward"
    input_field_optional: bool = False

    # PPO specific fields
    ppo_epochs: int = 4
    ppo_mini_batch_size: int = 8
    kl_coef: float = 0.1
    clip_range: float = 0.2


@dataclass
class RLConfig(Config):
    data: RLDataConfig

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert nested dictionaries to appropriate dataclasses
        data_config = RLDataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        resume_config = ResumeConfig(**config_dict.get('resume', {})) if 'resume' in config_dict else None

        return cls(
            name=config_dict.get('name', 'default'),
            overwrite=config_dict.get('overwrite', False),
            data=data_config,
            model=model_config,
            training=training_config,
            logging=logging_config,
            system=system_config,
            resume=resume_config
        )


class RLDataManager:
    """
    Data manager for reinforcement learning fine-tuning.
    Handles loading and processing of prompts, responses, and rewards.
    """
    def __init__(self, config: RLDataConfig, tokenizer: TokenizerManager, batch_size: int = 1):
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data = []
        self.validation_data = []
        self.max_context_size = config.preprocessing.get('max_context_size', 1024)
        self.hf_train_dataset = None
        self.hf_val_dataset = None
        self.using_hf_dataset = False

        # Check if we need to import datasets library
        if self.config.hf_dataset_name:
            try:
                from datasets import load_dataset
                self.load_dataset = load_dataset
                self.using_hf_dataset = True
            except ImportError:
                raise ImportError("The 'datasets' library is required for Hugging Face datasets. "
                                 "Please install it using: pip install datasets")

    def load_data(self):
        """Load training and validation data."""
        if self.using_hf_dataset:
            # Load data from Hugging Face dataset
            self._load_hf_dataset()
        else:
            # Load data from files
            if not self.config.input_file:
                raise ValueError("input_file must be specified when not using Hugging Face datasets")

            print(f"Loading RL data from {self.config.input_file}")
            self._load_file(self.config.input_file, self.data)

            if self.config.validation_file:
                print(f"Loading validation data from {self.config.validation_file}")
                self._load_file(self.config.validation_file, self.validation_data)

        print(f"Loaded {len(self.data)} training examples and {len(self.validation_data)} validation examples")

    def _load_hf_dataset(self):
        """Load data from Hugging Face dataset."""
        print(f"Loading Hugging Face dataset: {self.config.hf_dataset_name}")

        # Load the dataset
        dataset = self.load_dataset(
            self.config.hf_dataset_name,
            self.config.hf_dataset_config,
            streaming=self.config.use_streaming
        )

        # Check available splits
        available_splits = list(dataset.keys())
        print(f"Available splits: {available_splits}")

        # Verify train split exists
        if self.config.hf_train_split not in available_splits:
            raise ValueError(f"Training split '{self.config.hf_train_split}' not found. Available splits: {available_splits}")

        # Get training data
        train_dataset = dataset[self.config.hf_train_split]

        # Process training data
        if self.config.use_streaming:
            # For streaming datasets, we'll process examples on-the-fly
            self.hf_train_dataset = train_dataset
            # Load a small batch to initialize data for batch creation
            for i, example in enumerate(train_dataset):
                self._process_hf_example(example, self.data)
                if i >= 100:  # Load just enough examples to set up batches
                    break
        else:
            # For regular datasets, process all examples now
            for example in train_dataset:
                self._process_hf_example(example, self.data)

        # Get validation data if available
        if self.config.hf_val_split and self.config.hf_val_split in available_splits:
            val_dataset = dataset[self.config.hf_val_split]

            if self.config.use_streaming:
                # For streaming datasets, we'll process examples on-the-fly
                self.hf_val_dataset = val_dataset
                # Load a small batch to initialize validation_data for batch creation
                for i, example in enumerate(val_dataset):
                    self._process_hf_example(example, self.validation_data)
                    if i >= 100:  # Load just enough examples to set up batches
                        break
            else:
                # For regular datasets, process all examples now
                for example in val_dataset:
                    self._process_hf_example(example, self.validation_data)

    def _process_hf_example(self, example, data_list):
        """Process a single example from a Hugging Face dataset."""
        # Extract input, output, and reward
        input_text = example.get(self.config.input_field, "")
        output_text = example.get(self.config.output_field, "")
        reward = example.get(self.config.reward_field, 0.0)

        if not input_text and not self.config.input_field_optional:
            return

        # Format prompt with system prompt if provided
        prompt = ""
        if self.config.system_prompt:
            prompt += f"{self.config.system_prompt}\n\n"

        # Add formatted input
        prompt += self.config.prompt_format.format(instruction=input_text)

        # Format response
        response = self.config.response_format.format(response=output_text)

        # Tokenize prompt and response
        prompt_tokens = self.tokenizer.tokenize(prompt)
        response_tokens = self.tokenizer.tokenize(response)

        # Ensure we don't exceed max context size
        if len(prompt_tokens) + len(response_tokens) > self.max_context_size:
            # Truncate response if needed
            max_response_len = self.max_context_size - len(prompt_tokens)
            if max_response_len > 0:
                response_tokens = response_tokens[:max_response_len]
            else:
                # Skip if prompt alone exceeds max context
                return

        data_list.append({
            "prompt": prompt,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "reward": reward
        })

    def _load_file(self, file_path: str, data_list: list):
        """Load and process data from a JSONL file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)

                # Extract input, output, and reward
                input_text = example.get(self.config.input_field, "")
                output_text = example.get(self.config.output_field, "")
                reward = example.get(self.config.reward_field, 0.0)

                if not input_text and not self.config.input_field_optional:
                    continue

                # Format prompt with system prompt if provided
                prompt = ""
                if self.config.system_prompt:
                    prompt += f"{self.config.system_prompt}\n\n"

                # Add formatted input
                prompt += self.config.prompt_format.format(instruction=input_text)

                # Format response
                response = self.config.response_format.format(response=output_text)

                # Tokenize prompt and response
                prompt_tokens = self.tokenizer.tokenize(prompt)
                response_tokens = self.tokenizer.tokenize(response)

                # Ensure we don't exceed max context size
                if len(prompt_tokens) + len(response_tokens) > self.max_context_size:
                    # Truncate response if needed
                    max_response_len = self.max_context_size - len(prompt_tokens)
                    if max_response_len > 0:
                        response_tokens = response_tokens[:max_response_len]
                    else:
                        # Skip if prompt alone exceeds max context
                        continue

                data_list.append({
                    "prompt": prompt,
                    "response": response,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "reward": reward
                })

    def generate_batch(self, step: int):
        """Generate a batch of examples for training."""
        if self.using_hf_dataset and self.config.use_streaming and self.hf_train_dataset:
            # For streaming datasets, fetch examples on-the-fly
            batch_docs = []
            for _ in range(self.batch_size):
                # Get next example from the streaming dataset
                try:
                    example = next(iter(self.hf_train_dataset))
                    self._process_hf_example(example, batch_docs)
                except StopIteration:
                    # If we reach the end of the dataset, reset the iterator
                    self.hf_train_dataset = self.hf_train_dataset.shuffle()
                    example = next(iter(self.hf_train_dataset))
                    self._process_hf_example(example, batch_docs)

            return self._create_batch(batch_docs)
        else:
            # For regular datasets, use pre-loaded examples
            idx = (step * self.batch_size) % len(self.data)
            batch = self.data[idx:idx + self.batch_size]
            if len(batch) < self.batch_size:
                batch += self.data[:self.batch_size - len(batch)]
            return self._create_batch(batch)

    def generate_validation_batch(self, batch_idx: int):
        """Generate a batch of examples for validation."""
        if self.using_hf_dataset and self.config.use_streaming and self.hf_val_dataset:
            # For streaming datasets, fetch examples on-the-fly
            batch_docs = []
            for _ in range(self.batch_size):
                # Get next example from the streaming dataset
                try:
                    example = next(iter(self.hf_val_dataset))
                    self._process_hf_example(example, batch_docs)
                except StopIteration:
                    # If we reach the end of the dataset, reset the iterator
                    self.hf_val_dataset = self.hf_val_dataset.shuffle()
                    example = next(iter(self.hf_val_dataset))
                    self._process_hf_example(example, batch_docs)

            return self._create_batch(batch_docs)
        else:
            # For regular datasets, use pre-loaded examples
            if not self.validation_data:
                return None
            idx = (batch_idx * self.batch_size) % len(self.validation_data)
            batch = self.validation_data[idx:idx + self.batch_size]
            if len(batch) < self.batch_size:
                batch += self.validation_data[:self.batch_size - len(batch)]
            return self._create_batch(batch)

    def _create_batch(self, examples: list):
        """Create a batch from a list of examples."""
        if not examples:
            return None

        prompts = [ex["prompt_tokens"] for ex in examples]
        responses = [ex["response_tokens"] for ex in examples]
        rewards = [ex["reward"] for ex in examples]

        # Create input tensors for the model
        prompt_lengths = [len(p) for p in prompts]
        response_lengths = [len(r) for r in responses]

        # Pad sequences to the maximum length in the batch
        max_prompt_len = max(prompt_lengths)
        max_response_len = max(response_lengths)

        # Create padded arrays
        padded_prompts = np.zeros((len(prompts), max_prompt_len), dtype=np.int32)
        padded_responses = np.zeros((len(responses), max_response_len), dtype=np.int32)

        # Fill padded arrays
        for i, (p, r) in enumerate(zip(prompts, responses)):
            padded_prompts[i, :len(p)] = p
            padded_responses[i, :len(r)] = r

        return {
            "prompts": mx.array(padded_prompts),
            "responses": mx.array(padded_responses),
            "prompt_lengths": mx.array(prompt_lengths),
            "response_lengths": mx.array(response_lengths),
            "rewards": mx.array(rewards),
            "examples": examples  # Keep original examples for reference
        }

    def has_validation_data(self):
        """Check if validation data is available."""
        if self.using_hf_dataset:
            # For Hugging Face datasets, check if validation split is available
            return self.hf_val_dataset is not None or len(self.validation_data) > 0
        else:
            # For file-based datasets, check if validation data is loaded
            return len(self.validation_data) > 0

    def num_validation_batches(self):
        """Get the number of validation batches."""
        if not self.has_validation_data():
            return 0

        if self.using_hf_dataset and self.config.use_streaming and self.hf_val_dataset:
            # For streaming datasets, return a reasonable number of batches
            # This is an estimate since we don't know the exact size of the dataset
            return 100  # Arbitrary number, can be adjusted
        else:
            # For regular datasets, return the actual number of batches
            return max(1, len(self.validation_data) // self.batch_size)


class RewardModel:
    """
    A wrapper for the reward model used in RLHF.
    This can be a separate model or a function that computes rewards.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None

        if model_path:
            # Load external reward model
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load a reward model from a checkpoint."""
        # Implementation depends on the format of your reward model
        pass

    def compute_reward(self, prompts: mx.array, responses: mx.array) -> mx.array:
        """
        Compute rewards for the given prompts and responses.
        If no model is loaded, this should be overridden with a custom reward function.
        """
        if self.model:
            # Use the loaded model to compute rewards
            return self.model(prompts, responses)
        else:
            # Default implementation - should be overridden
            raise NotImplementedError(
                "No reward model loaded. Either provide a model_path or override this method."
            )


class RLTrainer:
    """
    Trainer for reinforcement learning fine-tuning using PPO.
    """
    def __init__(
        self, 
        config_path: str, 
        pretrained_model_path: Optional[str] = None,
        reward_model_path: Optional[str] = None
    ):
        self.config_path = config_path
        self.pretrained_model_path = pretrained_model_path
        self.reward_model_path = reward_model_path

        # Load configuration
        self.config = RLConfig.from_yaml(config_path)

        # Setup system
        self.setup_system()

        # Initialize components
        self.run_dir = None
        self.checkpoint_manager = None
        self.tokenizer = None
        self.data_manager = None
        self.model = None
        self.ref_model = None  # Reference model for KL penalty
        self.reward_model = None
        self.optimizer = None
        self.scheduler = None
        self.step = 0

        # Setup model, tokenizer, and data
        self.setup_model()
        self.setup_training()
        self.setup_logging()

    def setup_system(self):
        """Setup system configuration."""
        # Set random seed for reproducibility
        random.seed(self.config.system.seed)
        np.random.seed(self.config.system.seed)
        mx.random.seed(self.config.system.seed)

    def setup_model(self):
        """Setup model, tokenizer, and data."""
        # Setup run directory and checkpoint manager
        self.run_dir = CheckpointManager.setup_run_directory(self.config.name)
        self.checkpoint_manager = CheckpointManager()

        # Setup tokenizer
        if self.pretrained_model_path:
            # Use tokenizer from pretrained model
            tokenizer_path = Path(self.pretrained_model_path) / "tokenizer"
            if not tokenizer_path.exists():
                raise ValueError(f"Tokenizer not found at {tokenizer_path}")
            self.tokenizer = TokenizerManager(self.config.data, self.run_dir)
            self.tokenizer.use_external_tokenizer(str(tokenizer_path))
        elif self.config.data.tokenizer_path:
            # Use specified tokenizer
            self.tokenizer = TokenizerManager(self.config.data, self.run_dir)
            self.tokenizer.use_external_tokenizer(self.config.data.tokenizer_path)
        else:
            raise ValueError("No tokenizer specified. Provide either pretrained_model_path or data.tokenizer_path")

        # Setup data manager
        self.data_manager = RLDataManager(
            self.config.data,
            self.tokenizer,
            self.config.training.hyperparameters.batch_size
        )
        self.data_manager.load_data()

        # Load model architecture
        arch_module = importlib.import_module(f"arch.{self.config.model.architecture}")
        model_cls = getattr(arch_module, "Model")
        model_args_cls = getattr(arch_module, "ModelArgs")

        # Create model arguments
        model_args_dict = {}
        for category in ["dimensions", "attention", "normalization", "rope", "misc"]:
            if hasattr(self.config.model, category):
                category_dict = getattr(self.config.model, category)
                if category_dict:
                    for k, v in category_dict.items():
                        if v is not None:
                            model_args_dict[k] = v

        # Add vocabulary size
        model_args_dict["vocab_size"] = self.tokenizer.vocab_size

        # Create model arguments
        valid_args = filter_valid_args(model_args_cls, model_args_dict)
        model_args = model_args_cls(**valid_args)

        # Create model
        self.model = model_cls(model_args)

        # Create reference model (copy of the model for KL penalty)
        self.ref_model = model_cls(model_args)

        # Load pretrained weights if specified
        if self.pretrained_model_path:
            checkpoint_path = Path(self.pretrained_model_path) / "checkpoints" / "latest.safetensors"
            if not checkpoint_path.exists():
                # Try finding the best checkpoint
                checkpoint_path = Path(self.pretrained_model_path) / "checkpoints" / "best.safetensors"

            if not checkpoint_path.exists():
                raise ValueError(f"No checkpoint found at {checkpoint_path}")

            print(f"Loading pretrained weights from {checkpoint_path}")
            weights = mx.load(str(checkpoint_path))
            self.model.update(weights)
            self.ref_model.update(weights)  # Also update reference model

        # Setup reward model
        self.reward_model = RewardModel(self.reward_model_path)

    def setup_training(self):
        """Setup training components."""
        # Calculate total training steps
        if hasattr(self.config.training, "epochs") and self.config.training.epochs:
            num_examples = len(self.data_manager.data)
            batch_size = self.config.training.hyperparameters.batch_size
            steps_per_epoch = num_examples // batch_size
            total_steps = steps_per_epoch * self.config.training.epochs
        elif hasattr(self.config.training.hyperparameters, "iters"):
            total_steps = self.config.training.hyperparameters.iters
        else:
            raise ValueError("Either training.epochs or training.hyperparameters.iters must be specified")

        # Setup optimization manager
        self.optimization_manager = OptimizationManager(self.config.training, total_steps)

        # Create learning rate schedule
        self.scheduler = self.optimization_manager.create_scheduler()

        # Create optimizer
        self.optimizer = self.optimization_manager.create_optimizer(self.scheduler)

    def setup_logging(self):
        """Setup logging directories and metrics tracking."""
        # Create logging directory
        log_dir = self.run_dir / self.config.logging.log_dir
        log_dir.mkdir(exist_ok=True)

        # Create checkpoint directory
        checkpoint_dir = self.run_dir / self.config.logging.checkpoint_dir
        checkpoint_dir.mkdir(exist_ok=True)

        # Save configuration
        config_path = self.run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config.__dict__, f)

        # Initialize metrics
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "rewards": [],
            "kl_divergence": [],
            "policy_loss": [],
            "value_loss": [],
            "steps": [],
            "tokens_per_second": [],
            "learning_rate": [],
            "total_tokens": 0
        }

        # Save initial metrics
        self.save_metrics()

    def compute_policy_loss(self, logprobs, old_logprobs, advantages, clip_range):
        """Compute PPO policy loss."""
        ratio = mx.exp(logprobs - old_logprobs)
        clipped_ratio = mx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -mx.minimum(ratio * advantages, clipped_ratio * advantages)
        return policy_loss.mean()

    def compute_value_loss(self, values, returns, clip_range):
        """Compute PPO value loss."""
        clipped_values = values + mx.clip(returns - values, -clip_range, clip_range)
        value_loss1 = (values - returns) ** 2
        value_loss2 = (clipped_values - returns) ** 2
        value_loss = 0.5 * mx.maximum(value_loss1, value_loss2)
        return value_loss.mean()

    def compute_kl_divergence(self, logprobs, ref_logprobs):
        """Compute KL divergence between policy and reference policy."""
        return (mx.exp(logprobs) * (logprobs - ref_logprobs)).mean()

    def generate_responses(self, prompts, max_length=100):
        """Generate responses for the given prompts."""
        # Implementation depends on your model's generation function
        pass

    def train_ppo_epoch(self, batch):
        """Train one PPO epoch on a batch of data."""
        prompts = batch["prompts"]
        prompt_lengths = batch["prompt_lengths"]
        old_responses = batch["responses"]
        old_response_lengths = batch["response_lengths"]
        rewards = batch["rewards"]

        # Generate new responses with the current policy
        new_responses, new_logprobs = self.generate_responses(prompts)

        # Compute rewards for new responses
        new_rewards = self.reward_model.compute_reward(prompts, new_responses)

        # Compute advantages (new_rewards - old_rewards)
        advantages = new_rewards - rewards

        # Get logprobs from reference model for KL penalty
        with mx.stop_gradient():
            ref_logprobs = self.ref_model.compute_logprobs(prompts, new_responses)

        # Compute losses
        policy_loss = self.compute_policy_loss(
            new_logprobs, 
            old_logprobs, 
            advantages, 
            self.config.data.clip_range
        )

        kl_div = self.compute_kl_divergence(new_logprobs, ref_logprobs)
        kl_loss = self.config.data.kl_coef * kl_div

        # Total loss
        loss = policy_loss + kl_loss

        # Compute gradients and update model
        grads = mx.grad(self.model, loss)
        self.optimizer.update(self.model, grads)

        return {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "rewards": new_rewards.mean().item(),
            "loss": loss.item()
        }

    def train(self):
        """Train the model using PPO."""
        print(f"Starting RL training for {self.config.name}")

        # Training loop
        step = 0
        total_tokens = 0
        best_val_loss = float('inf')

        # Main training loop
        while True:
            start_time = time.time()

            # Get batch
            batch = self.data_manager.generate_batch(step)

            # Train multiple PPO epochs on this batch
            epoch_metrics = {
                "policy_loss": 0,
                "kl_div": 0,
                "rewards": 0,
                "loss": 0
            }

            for _ in range(self.config.data.ppo_epochs):
                metrics = self.train_ppo_epoch(batch)
                for k, v in metrics.items():
                    epoch_metrics[k] += v / self.config.data.ppo_epochs

            # Update learning rate
            lr = self.scheduler(step)

            # Calculate tokens processed
            tokens_in_batch = batch["prompt_lengths"].sum().item() + batch["response_lengths"].sum().item()
            total_tokens += tokens_in_batch

            # Log metrics
            if step % self.config.logging.steps['logging_interval'] == 0:
                self.log_metrics(
                    step=step,
                    loss=epoch_metrics["loss"],
                    tokens=tokens_in_batch,
                    total_tokens=total_tokens,
                    start_time=start_time,
                    rewards=epoch_metrics["rewards"],
                    policy_loss=epoch_metrics["policy_loss"],
                    kl_div=epoch_metrics["kl_div"]
                )

            # Validation
            if (
                step % self.config.logging.steps['validation_interval'] == 0 and
                self.data_manager.has_validation_data()
            ):
                val_metrics = self.validate()
                val_loss = val_metrics["loss"]

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint("best", val_loss)

                # Log validation metrics
                self.log_metrics(
                    step=step,
                    loss=epoch_metrics["loss"],
                    tokens=tokens_in_batch,
                    total_tokens=total_tokens,
                    start_time=start_time,
                    rewards=epoch_metrics["rewards"],
                    policy_loss=epoch_metrics["policy_loss"],
                    kl_div=epoch_metrics["kl_div"],
                    val_loss=val_loss,
                    val_rewards=val_metrics["rewards"]
                )

            # Save checkpoint
            if step % self.config.logging.steps['checkpoint_interval'] == 0:
                self.save_checkpoint("latest")

            # Check if training is complete
            if hasattr(self.config.training.hyperparameters, "iters") and step >= self.config.training.hyperparameters.iters:
                break

            if (
                hasattr(self.config.training, "epochs") and
                step >= (self.config.training.epochs * len(self.data_manager.data) // self.config.training.hyperparameters.batch_size)
            ):
                break

            step += 1

        # Save final checkpoint
        self.save_checkpoint("final")
        print(f"Training complete. Model saved to {self.run_dir}")

    def validate(self):
        """Validate the model on the validation set."""
        if not self.data_manager.has_validation_data():
            return {"loss": 0.0, "rewards": 0.0}

        total_loss = 0.0
        total_rewards = 0.0
        num_batches = min(self.data_manager.num_validation_batches(), 10)  # Limit validation batches

        for i in range(num_batches):
            batch = self.data_manager.generate_validation_batch(i)

            # Generate responses
            with mx.stop_gradient():
                responses, logprobs = self.generate_responses(batch["prompts"])
                rewards = self.reward_model.compute_reward(batch["prompts"], responses)

                # Compute reference logprobs
                ref_logprobs = self.ref_model.compute_logprobs(batch["prompts"], responses)

                # Compute KL divergence
                kl_div = self.compute_kl_divergence(logprobs, ref_logprobs)

                # Compute loss (negative reward + KL penalty)
                loss = -rewards.mean() + self.config.data.kl_coef * kl_div

            total_loss += loss.item()
            total_rewards += rewards.mean().item()

        return {
            "loss": total_loss / num_batches,
            "rewards": total_rewards / num_batches
        }

    def save_checkpoint(self, step: int | str, val_loss: float = None):
        """Save a checkpoint of the model."""
        checkpoint_dir = self.run_dir / self.config.logging.checkpoint_dir

        if isinstance(step, int):
            checkpoint_path = checkpoint_dir / f"step_{step}.safetensors"
        else:
            checkpoint_path = checkpoint_dir / f"{step}.safetensors"

        # Save model weights
        mx.save(str(checkpoint_path), self.model.parameters())

        # Save optimizer state
        optimizer_path = checkpoint_path.with_suffix(".optimizer.safetensors")
        mx.save(str(optimizer_path), self.optimizer.state)

        # Save metadata
        metadata = {
            "step": self.step,
            "timestamp": datetime.now().isoformat(),
            "total_tokens": self.metrics["total_tokens"]
        }

        if val_loss is not None:
            metadata["val_loss"] = val_loss

        metadata_path = checkpoint_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        print(f"Saved checkpoint to {checkpoint_path}")

    def log_metrics(
        self, 
        step: int, 
        loss: float, 
        tokens: int, 
        total_tokens: int, 
        start_time: float,
        rewards: float = None,
        policy_loss: float = None,
        kl_div: float = None,
        val_loss: float = None,
        val_rewards: float = None
    ):
        """Log training metrics."""
        elapsed = time.time() - start_time
        tokens_per_second = tokens / elapsed if elapsed > 0 else 0

        # Update metrics
        self.metrics["steps"].append(step)
        self.metrics["train_loss"].append(loss)
        self.metrics["tokens_per_second"].append(tokens_per_second)
        self.metrics["learning_rate"].append(self.scheduler(step))
        self.metrics["total_tokens"] = total_tokens

        if rewards is not None:
            self.metrics["rewards"].append(rewards)

        if policy_loss is not None:
            self.metrics["policy_loss"].append(policy_loss)

        if kl_div is not None:
            self.metrics["kl_divergence"].append(kl_div)

        if val_loss is not None:
            self.metrics["val_loss"].append(val_loss)

        # Print metrics
        log_str = f"Step {step}: loss={loss:.4f}, "

        if rewards is not None:
            log_str += f"rewards={rewards:.4f}, "

        if policy_loss is not None:
            log_str += f"policy_loss={policy_loss:.4f}, "

        if kl_div is not None:
            log_str += f"kl_div={kl_div:.4f}, "

        if val_loss is not None:
            log_str += f"val_loss={val_loss:.4f}, "

        if val_rewards is not None:
            log_str += f"val_rewards={val_rewards:.4f}, "

        log_str += f"tokens/s={tokens_per_second:.1f}, lr={self.scheduler(step):.6f}"
        print(log_str)

        # Save metrics to file
        self.save_metrics()

    def save_metrics(self):
        """Save metrics to a JSON file."""
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f)


def main():
    parser = argparse.ArgumentParser(description="Train a model with reinforcement learning")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--pretrained_model", help="Path to the pretrained model directory")
    parser.add_argument("--reward_model", help="Path to the reward model")
    args = parser.parse_args()

    trainer = RLTrainer(args.config, args.pretrained_model, args.reward_model)
    trainer.train()


if __name__ == "__main__":
    main()
