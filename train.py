import json
import random
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
#from mlx_lm.models.llama import Model, ModelArgs
import importlib
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import inspect

def filter_valid_args(cls, arg_dict):
    valid_params = inspect.signature(cls).parameters
    return {k: v for k, v in arg_dict.items() if k in valid_params}


@dataclass
class DataConfig:
    input_file: Optional[str] = None
    preprocessing: Dict[str, int] = None
    tokenizer: Dict[str, Any] = None
    tokenizer_path: Optional[str] = None  # Path to a directory containing a tokenizer.json file
    validation_file: Optional[str] = None
    weight_path: Optional[str] = None
    # Fields for vision-language models
    is_vision_language: bool = False  # Flag to indicate if this is a vision-language model
    image_field: Optional[str] = None  # Field name for image paths in the input data
    text_field: Optional[str] = "text"  # Field name for text in the input data
    image_processor_path: Optional[str] = None  # Path to image processor for vision-language models
    max_images_per_sample: int = 1  # Maximum number of images per sample
    # Fields for Hugging Face datasets
    hf_dataset_name: Optional[str] = None  # Name of the Hugging Face dataset
    hf_dataset_config: Optional[str] = None  # Configuration name for the dataset
    hf_train_split: Optional[str] = "train"  # Training split name
    hf_val_split: Optional[str] = "validation"  # Validation split name
    use_streaming: bool = False  # Whether to use streaming mode for the dataset
    # Multimodal loss configuration
    # Example configuration:
    # multimodal_loss:
    #   lm_loss_weight: 1.0                # Weight for language modeling loss (default: 1.0)
    #   enable_contrastive_loss: true      # Enable contrastive loss (default: false)
    #   contrastive_loss_weight: 0.5       # Weight for contrastive loss (default: 0.5)
    #   contrastive_loss_temperature: 0.07 # Temperature for contrastive loss (default: 0.07)
    #   enable_matching_loss: true         # Enable image-text matching loss (default: false)
    #   matching_loss_weight: 0.5          # Weight for matching loss (default: 0.5)
    #   verbose_loss_logging: true         # Enable verbose logging of loss components (default: false)
    multimodal_loss: Optional[Dict[str, Any]] = None  # Configuration for multimodal loss functions
    # Vision-language evaluation metrics configuration
    # Example configuration:
    # vlm_evaluation:
    #   enable_retrieval_metrics: true     # Enable image-text retrieval metrics (default: false)
    #   retrieval_k_values: [1, 5, 10]     # K values for Recall@K metrics (default: [1, 5, 10])
    #   enable_vqa_metrics: false          # Enable visual question answering metrics (default: false)
    #   enable_captioning_metrics: false   # Enable image captioning metrics (default: false)
    #   enable_classification_metrics: false # Enable zero-shot classification metrics (default: false)
    #   classification_categories: []      # Categories for zero-shot classification (default: [])
    #   verbose_metrics_logging: true      # Enable verbose logging of metrics (default: false)
    vlm_evaluation: Optional[Dict[str, Any]] = None  # Configuration for vision-language evaluation metrics

@dataclass
class ModelConfig:
    architecture: str
    dimensions: Dict[str, int]
    attention: Dict[str, Any]
    normalization: Dict[str, float]
    rope: Dict[str, Any]
    misc: Dict[str, bool]
    lora: Optional[Dict[str, Any]] = None  # LoRA/QLoRA parameters for fine-tuning
    model_source: Optional[str] = None  # Source of the model: "custom", "mlx_lm", or "mlx_vlm"

@dataclass
class TrainingConfig:
    hyperparameters: Dict[str, Any]
    scheduler: Dict[str, Any]
    optimization: Dict[str, Any]
    epochs: Optional[int] = None

@dataclass
class LoggingConfig:
    log_dir: str
    checkpoint_dir: str
    steps: Dict[str, int]
    metrics: Dict[str, bool]
    # Default to 0 (no validation) if not specified

@dataclass
class SystemConfig:
    seed: int
    device: str

@dataclass
class ResumeConfig:
    checkpoint: str  # Path to checkpoint base name
    reset_optimizer: bool = False  # Optional flag to reset optimizer state

@dataclass
class Config:
    name: str  # New field for run name
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    system: SystemConfig
    resume: Optional[ResumeConfig] = None
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Validate that name is present
        if 'name' not in config_dict:
            raise ValueError("Config must specify a 'name' field at the top level")

        # Extract epochs if it exists at the top level of training config
        training_config = config_dict['training'].copy()
        epochs = training_config.pop('epochs', None)

        # Extract resume config if present
        resume_config = None
        if 'resume' in config_dict:
            resume_config = ResumeConfig(**config_dict['resume'])

        return cls(
            name=config_dict['name'],
            overwrite=config_dict.get('overwrite', False),
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**training_config, epochs=epochs),
            logging=LoggingConfig(**config_dict['logging']),
            system=SystemConfig(**config_dict['system']),
            resume=resume_config
        )

class VLMEvaluator:
    """Evaluator for vision-language models.

    This class computes various metrics for vision-language models, including:
    - Image-text retrieval metrics (Recall@K)
    - Visual question answering accuracy (if applicable)
    - Image captioning metrics (if applicable)
    - Zero-shot classification accuracy (if applicable)
    """

    def __init__(self, config: Dict[str, Any], tokenizer):
        """Initialize the VLMEvaluator.

        Args:
            config: Configuration dictionary for vision-language evaluation
            tokenizer: Tokenizer for text processing
        """
        self.config = config or {}
        self.tokenizer = tokenizer
        self.metrics = {}

        # Set default values for configuration options
        self.enable_retrieval_metrics = self.config.get("enable_retrieval_metrics", False)
        self.retrieval_k_values = self.config.get("retrieval_k_values", [1, 5, 10])
        self.enable_vqa_metrics = self.config.get("enable_vqa_metrics", False)
        self.enable_captioning_metrics = self.config.get("enable_captioning_metrics", False)
        self.enable_classification_metrics = self.config.get("enable_classification_metrics", False)
        self.classification_categories = self.config.get("classification_categories", [])
        self.verbose_metrics_logging = self.config.get("verbose_metrics_logging", False)

    def compute_metrics(self, model, batch, model_output):
        """Compute metrics for a batch of data.

        Args:
            model: The vision-language model
            batch: Batch of data (dictionary with text_tokens and images)
            model_output: Output from the model

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Extract text tokens and images from batch
        text_tokens = batch["text_tokens"]
        images = batch["images"]

        # Extract embeddings from model output if available
        image_embeddings = None
        text_embeddings = None

        if isinstance(model_output, dict):
            image_embeddings = model_output.get("image_embeddings", None)
            text_embeddings = model_output.get("text_embeddings", None)

        # Compute retrieval metrics if enabled and embeddings are available
        if self.enable_retrieval_metrics and image_embeddings is not None and text_embeddings is not None:
            retrieval_metrics = self._compute_retrieval_metrics(image_embeddings, text_embeddings)
            metrics.update(retrieval_metrics)

        # Compute VQA metrics if enabled
        if self.enable_vqa_metrics:
            vqa_metrics = self._compute_vqa_metrics(model, batch, model_output)
            metrics.update(vqa_metrics)

        # Compute captioning metrics if enabled
        if self.enable_captioning_metrics:
            captioning_metrics = self._compute_captioning_metrics(model, batch, model_output)
            metrics.update(captioning_metrics)

        # Compute classification metrics if enabled
        if self.enable_classification_metrics and len(self.classification_categories) > 0:
            classification_metrics = self._compute_classification_metrics(model, batch, model_output)
            metrics.update(classification_metrics)

        return metrics

    def _compute_retrieval_metrics(self, image_embeddings, text_embeddings):
        """Compute image-text retrieval metrics.

        Args:
            image_embeddings: Image embeddings from the model
            text_embeddings: Text embeddings from the model

        Returns:
            Dictionary of retrieval metrics
        """
        metrics = {}

        # Normalize embeddings
        image_embeddings = image_embeddings / (mx.linalg.norm(image_embeddings, axis=-1, keepdims=True) + 1e-8)
        text_embeddings = text_embeddings / (mx.linalg.norm(text_embeddings, axis=-1, keepdims=True) + 1e-8)

        # Compute similarity matrix
        similarity = mx.matmul(image_embeddings, text_embeddings.transpose())

        # Compute image-to-text retrieval metrics
        i2t_ranks = self._compute_ranks(similarity)
        t2i_ranks = self._compute_ranks(similarity.transpose())

        # Compute Recall@K for each K value
        for k in self.retrieval_k_values:
            # Image-to-text retrieval
            i2t_recall = (i2t_ranks < k).sum() / len(i2t_ranks)
            metrics[f"i2t_recall@{k}"] = float(i2t_recall)

            # Text-to-image retrieval
            t2i_recall = (t2i_ranks < k).sum() / len(t2i_ranks)
            metrics[f"t2i_recall@{k}"] = float(t2i_recall)

            # Mean recall
            mean_recall = (float(i2t_recall) + float(t2i_recall)) / 2
            metrics[f"mean_recall@{k}"] = mean_recall

        return metrics

    def _compute_ranks(self, similarity):
        """Compute ranks from similarity matrix.

        Args:
            similarity: Similarity matrix

        Returns:
            Array of ranks
        """
        # Get indices of sorted similarities (descending order)
        sorted_indices = mx.argsort(-similarity, axis=1)

        # Get the position of the correct match (diagonal elements)
        ranks = mx.zeros(similarity.shape[0], dtype=mx.int32)
        for i in range(similarity.shape[0]):
            # Find the position of i in the sorted indices for row i
            for j in range(similarity.shape[1]):
                if sorted_indices[i, j] == i:
                    ranks[i] = j
                    break

        return ranks

    def _compute_vqa_metrics(self, model, batch, model_output):
        """Compute visual question answering metrics.

        This is a placeholder for VQA metrics. In a real implementation, this would
        compute accuracy based on ground truth answers.

        Returns:
            Dictionary of VQA metrics
        """
        # Placeholder for VQA metrics
        return {"vqa_accuracy": 0.0}

    def _compute_captioning_metrics(self, model, batch, model_output):
        """Compute image captioning metrics.

        This is a placeholder for captioning metrics. In a real implementation, this would
        compute BLEU, METEOR, CIDEr, etc. based on ground truth captions.

        Returns:
            Dictionary of captioning metrics
        """
        # Placeholder for captioning metrics
        return {"captioning_bleu": 0.0}

    def _compute_classification_metrics(self, model, batch, model_output):
        """Compute zero-shot classification metrics.

        This is a placeholder for classification metrics. In a real implementation, this would
        compute accuracy based on ground truth labels.

        Returns:
            Dictionary of classification metrics
        """
        # Placeholder for classification metrics
        return {"classification_accuracy": 0.0}

    def aggregate_metrics(self, batch_metrics):
        """Aggregate metrics across batches.

        Args:
            batch_metrics: List of metric dictionaries from each batch

        Returns:
            Dictionary of aggregated metrics
        """
        if not batch_metrics:
            return {}

        # Initialize aggregated metrics
        aggregated = {}

        # Get all metric keys
        all_keys = set()
        for metrics in batch_metrics:
            all_keys.update(metrics.keys())

        # Aggregate metrics across batches
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in batch_metrics if key in metrics]
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated

    def format_metrics(self, metrics):
        """Format metrics for logging.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Formatted string of metrics
        """
        if not metrics:
            return ""

        # Format each metric
        formatted = []
        for key, value in metrics.items():
            formatted.append(f"{key}={value:.4f}")

        return " | ".join(formatted)


class CheckpointManager:
    @staticmethod
    def validate_unique_name(name: str) -> None:
        """Validates that the run directory doesn't already exist"""
        run_path = Path('runs') / name
        if run_path.exists():
            raise ValueError(f"Run directory already exists for name '{name}'")

    @staticmethod
    def setup_run_directory(name: str) -> tuple[Path, Path, Path]:
        """Creates and returns paths for run directory structure"""
        run_dir = Path('runs') / name
        checkpoint_dir = run_dir / 'checkpoints'

        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(exist_ok=True)

        return run_dir, run_dir / 'log.txt', checkpoint_dir

    @staticmethod
    def get_checkpoint_paths(checkpoint_path: str) -> tuple[str, str, str]:
        """Returns the paths for model, optimizer, and state files"""
        model_path = f"{checkpoint_path}_model.safetensors"
        optimizer_path = f"{checkpoint_path}_optimizer.safetensors"
        state_path = f"{checkpoint_path}_state.json"
        return model_path, optimizer_path, state_path

class TokenizerManager:
    def __init__(self, config: DataConfig, run_dir: Optional[Path] = None):
        self.config = config
        self.external_tokenizer = None

        # Check if an external tokenizer path is provided
        if config.tokenizer_path is not None:
            self.use_external_tokenizer(config.tokenizer_path)

            # If we have a run directory, copy the tokenizer to it
            if run_dir is not None:
                self.copy_tokenizer_to_run_dir(config.tokenizer_path, run_dir)
        else:
            # Fall back to byte-level tokenization
            self.setup_vocabulary()

    def use_external_tokenizer(self, tokenizer_path: str):
        """Load and use an external tokenizer from the specified path."""
        from tokenizers import Tokenizer
        import os
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")

        if not os.path.exists(tokenizer_file):
            raise ValueError(f"Tokenizer file not found at {tokenizer_file}")

        print(f"Loading external tokenizer from {tokenizer_file}")
        self.external_tokenizer = Tokenizer.from_file(tokenizer_file)

        # Extract special token IDs
        vocab = self.external_tokenizer.get_vocab()
        special_tokens = self.config.tokenizer['special_tokens']

        # Map special tokens to their IDs
        self.PAD_TOKEN = vocab.get(special_tokens['pad'])
        self.BOS_TOKEN = vocab.get(special_tokens['bos'])
        self.EOS_TOKEN = vocab.get(special_tokens['eos'])
        self.VOCAB_SIZE = len(vocab)

        if self.PAD_TOKEN is None or self.BOS_TOKEN is None or self.EOS_TOKEN is None:
            raise ValueError(f"One or more special tokens not found in the external tokenizer vocabulary")

    def copy_tokenizer_to_run_dir(self, tokenizer_path: str, run_dir: Path):
        """Copy the tokenizer files to the run directory."""
        import shutil
        import os

        # Create tokenizer directory in run_dir
        run_tokenizer_dir = run_dir / 'tokenizer'
        os.makedirs(run_tokenizer_dir, exist_ok=True)

        # Copy tokenizer.json
        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        shutil.copy2(tokenizer_file, run_tokenizer_dir / "tokenizer.json")

        print(f"Copied tokenizer to {run_tokenizer_dir}")

    def setup_vocabulary(self):
        """Set up the byte-level tokenization vocabulary."""
        normal_vocab_size = self.config.tokenizer['normal_vocab_size']
        special_tokens = self.config.tokenizer['special_tokens']

        # Create vocabulary mapping
        self.special_token_map = {
            token: normal_vocab_size + idx 
            for idx, token in enumerate(special_tokens.values())
        }

        # Store common tokens
        self.PAD_TOKEN = self.special_token_map[special_tokens['pad']]
        self.BOS_TOKEN = self.special_token_map[special_tokens['bos']]
        self.EOS_TOKEN = self.special_token_map[special_tokens['eos']]
        self.VOCAB_SIZE = normal_vocab_size + len(self.special_token_map)

    def tokenize(self, text: str) -> list:
        if self.external_tokenizer is not None:
            # Use external tokenizer
            encoded = self.external_tokenizer.encode(text)
            return encoded.ids
        else:
            # Use byte-level tokenization
            return list(text.encode('utf-8'))

    def detokenize(self, tokens: list) -> str:
        if self.external_tokenizer is not None:
            # Use external tokenizer
            return self.external_tokenizer.decode(tokens.tolist())
        else:
            # Use byte-level detokenization
            return bytes(tokens).decode('utf-8', errors='ignore')

    def tokenize_doc(self, doc: str) -> list:
        """Tokenize a document, ensuring it doesn't exceed the max context size.

        Args:
            doc: The text to tokenize

        Returns:
            A list of token IDs, including BOS and EOS tokens
        """
        max_length = self.config.preprocessing['max_context_size']

        if self.external_tokenizer is not None:
            # Use external tokenizer
            encoded = self.external_tokenizer.encode(doc)
            tokens = encoded.ids[:max_length]
            return [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        else:
            # Use byte-level tokenization
            return [self.BOS_TOKEN] + self.tokenize(doc)[:max_length] + [self.EOS_TOKEN]

class DataManager:
    def __init__(self, config: DataConfig, tokenizer: TokenizerManager, batch_size: int = 1):
        self.config = config
        self.tokenizer = tokenizer
        self.train_docs = []
        self.val_docs = []
        self.train_idx = None
        self.val_idx = None
        self.batch_size = batch_size
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

        # Initialize image processor for vision-language models
        self.image_processor = None
        if self.config.is_vision_language:
            if self.config.image_field is None:
                raise ValueError("image_field must be specified for vision-language models")

            # Try to import mlx_vlm for image processing
            try:
                import mlx_vlm
                from mlx_vlm.utils import load_processor

                # Load image processor if path is provided
                if self.config.image_processor_path:
                    print(f"Loading image processor from {self.config.image_processor_path}")
                    self.image_processor = load_processor(self.config.image_processor_path)
                else:
                    print("No image processor path provided, will use default processing")
            except ImportError:
                raise ImportError("mlx_vlm is required for vision-language models. Please install it with 'pip install mlx_vlm'")

        self.load_data()

    def load_data(self):
        if self.using_hf_dataset:
            # Load data from Hugging Face dataset
            self._load_hf_dataset()
        else:
            # Load data from files
            if not self.config.input_file:
                raise ValueError("input_file must be specified when not using Hugging Face datasets")

            # Load training data
            self._load_file(self.config.input_file, self.train_docs)

            # Load validation data if specified
            if self.config.validation_file:
                self._load_file(self.config.validation_file, self.val_docs)

        # Set up training batches
        self.train_idx = sorted(range(len(self.train_docs)), key=lambda idx: len(self.train_docs[idx]))
        random.shuffle(self.train_idx)
        self.train_batch_idx = [
            self.train_idx[i : i + self.batch_size : 1]
            for i in range(0, len(self.train_idx) - self.batch_size + 1, self.batch_size)
        ]
        self.train_indices = np.random.permutation(len(self.train_batch_idx))

        # Set up validation batches if we have validation data
        if self.val_docs:
            self.val_idx = sorted(range(len(self.val_docs)), key=lambda idx: len(self.val_docs[idx]))
            self.val_batch_idx = [
                self.val_idx[i : i + self.batch_size : 1]
                for i in range(0, len(self.val_idx) - self.batch_size + 1, self.batch_size)
            ]
            self.val_indices = np.random.permutation(len(self.val_batch_idx))
            self.val_ptr = 0  # Pointer for validation batches

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
            # Load a small batch to initialize train_docs for batch creation
            for i, example in enumerate(train_dataset):
                self._process_hf_example(example, self.train_docs)
                if i >= 100:  # Load just enough examples to set up batches
                    break
        else:
            # For regular datasets, process all examples now
            for example in train_dataset:
                self._process_hf_example(example, self.train_docs)

        # Get validation data if available
        if self.config.hf_val_split and self.config.hf_val_split in available_splits:
            val_dataset = dataset[self.config.hf_val_split]

            if self.config.use_streaming:
                # For streaming datasets, we'll process examples on-the-fly
                self.hf_val_dataset = val_dataset
                # Load a small batch to initialize val_docs for batch creation
                for i, example in enumerate(val_dataset):
                    self._process_hf_example(example, self.val_docs)
                    if i >= 100:  # Load just enough examples to set up batches
                        break
            else:
                # For regular datasets, process all examples now
                for example in val_dataset:
                    self._process_hf_example(example, self.val_docs)

        print(f"Loaded {len(self.train_docs)} training examples and {len(self.val_docs)} validation examples")

    def _process_hf_example(self, example, docs_list):
        """Process a single example from a Hugging Face dataset."""
        # Handle vision-language data
        if self.config.is_vision_language:
            # Get text from the specified field
            text = example.get(self.config.text_field, "")

            # Get image paths from the specified field
            images = example.get(self.config.image_field, [])

            # Ensure images is a list
            if not isinstance(images, list):
                images = [images]

            # Limit number of images if needed
            images = images[:self.config.max_images_per_sample]

            # Create a sample with text and images
            sample = {
                "text": text,
                "images": images
            }

            # Add the sample to the docs list
            docs_list.append(sample)
        else:
            # Handle text-only data
            text = example.get(self.config.text_field, "")
            if not text:
                return

            chunk_size = self.config.preprocessing['max_context_size']
            overlap = self.config.preprocessing.get('chunk_overlap', 0)

            # Handle overlapping chunks if specified
            stride = chunk_size - overlap
            for i in range(0, len(text), stride):
                chunk_text = text[i : i + chunk_size]
                docs_list.append(chunk_text)

    def _load_file(self, file_path: str, docs_list: list):
        """Helper method to load documents from a file."""
        with open(file_path, 'r') as f:
            for line in f:
                d = json.loads(line)

                # Handle vision-language data
                if self.config.is_vision_language:
                    # Get text from the specified field
                    text = d.get(self.config.text_field, "")

                    # Get image paths from the specified field
                    images = d.get(self.config.image_field, [])

                    # Ensure images is a list
                    if not isinstance(images, list):
                        images = [images]

                    # Limit number of images if needed
                    images = images[:self.config.max_images_per_sample]

                    # Create a sample with text and images
                    sample = {
                        "text": text,
                        "images": images
                    }

                    # Add the sample to the docs list
                    docs_list.append(sample)
                else:
                    # Handle text-only data (original behavior)
                    text = d.get(self.config.text_field, d.get("text", ""))
                    chunk_size = self.config.preprocessing['max_context_size']
                    overlap = self.config.preprocessing.get('chunk_overlap', 0)

                    # Handle overlapping chunks if specified
                    stride = chunk_size - overlap
                    for i in range(0, len(text), stride):
                        chunk_text = text[i : i + chunk_size]
                        docs_list.append(chunk_text)

    def generate_batch(self, step: int) -> mx.array:
        """Generate a training batch."""
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
            indices = self.train_batch_idx[self.train_indices[step % len(self.train_indices)]]
            return self._create_batch([self.train_docs[i] for i in indices])

    def generate_validation_batch(self, batch_idx: int) -> mx.array:
        """Generate a validation batch."""
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
            if not self.val_docs or batch_idx >= len(self.val_batch_idx):
                raise ValueError("No validation data available or batch index out of range")

            indices = self.val_batch_idx[self.val_indices[self.val_ptr % len(self.val_indices)]]
            self.val_ptr += 1
            return self._create_batch([self.val_docs[i] for i in indices])

    def _create_batch(self, docs: list) -> Union[mx.array, Dict[str, Any]]:
        """Helper method to create and pad a batch from documents.

        For text-only models, returns an mx.array of token IDs.
        For vision-language models, returns a dictionary with 'text_tokens' and 'images'.
        """
        if self.config.is_vision_language:
            # Process text
            text_batch = [self.tokenizer.tokenize_doc(doc["text"]) for doc in docs]
            max_len = max(len(x) for x in text_batch)

            # Pad sequences
            for i in range(len(text_batch)):
                text_batch[i] += [self.tokenizer.PAD_TOKEN] * (max_len - len(text_batch[i]))

            # Process images
            image_batch = []
            for doc in docs:
                # Process each image in the document
                doc_images = []
                for image_path in doc["images"]:
                    # Process the image
                    processed_image = self._process_image(image_path)
                    if processed_image is not None:
                        doc_images.append(processed_image)

                # If no images were successfully processed, add None
                if not doc_images:
                    doc_images = [None]

                image_batch.append(doc_images)

            # Return a dictionary with text tokens and images
            return {
                "text_tokens": mx.array(text_batch),
                "images": image_batch
            }
        else:
            # Text-only processing (original behavior)
            batch = [self.tokenizer.tokenize_doc(doc) for doc in docs]
            max_len = max(len(x) for x in batch)

            # Pad sequences
            for i in range(len(batch)):
                batch[i] += [self.tokenizer.PAD_TOKEN] * (max_len - len(batch[i]))

            return mx.array(batch)

    def _process_image(self, image_path: str) -> Any:
        """Process an image for vision-language models.

        Args:
            image_path: Path to the image file

        Returns:
            Processed image or None if processing failed
        """
        try:
            # Use mlx_vlm's image processing if available
            if self.image_processor is not None:
                # Load and process the image using the processor
                return self.image_processor(image_path)
            else:
                # Default image processing using mlx_vlm utilities
                try:
                    from mlx_vlm.utils import load_and_process_image
                    return load_and_process_image(image_path)
                except (ImportError, Exception) as e:
                    print(f"Error processing image {image_path}: {e}")
                    return None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    @property
    def has_validation_data(self) -> bool:
        """Check if validation data is available."""
        if self.using_hf_dataset:
            # For Hugging Face datasets, check if validation split is available
            return self.hf_val_dataset is not None or len(self.val_docs) > 0
        else:
            # For file-based datasets, check if validation file is specified and loaded
            return self.config.validation_file is not None and len(self.val_docs) > 0

    @property
    def num_validation_batches(self) -> int:
        """Get the number of validation batches."""
        if not self.has_validation_data:
            return 0

        if self.using_hf_dataset and self.config.use_streaming and self.hf_val_dataset:
            # For streaming datasets, return a reasonable number of batches
            # This is an estimate since we don't know the exact size of the dataset
            return 100  # Arbitrary number, can be adjusted
        else:
            # For regular datasets, return the actual number of batches
            return len(self.val_batch_idx) if hasattr(self, 'val_batch_idx') else 0

class OptimizationManager:
    def __init__(self, config: TrainingConfig, num_training_steps: int):
        self.config = config
        self.num_training_steps = num_training_steps

    def create_scheduler(self) -> Any:
        cfg = self.config.scheduler
        initial_lr = self.config.hyperparameters['learning_rate']

        if cfg['type'] == 'cosine_with_warmup':
            warmup = optim.linear_schedule(0, initial_lr, steps=cfg['warmup_steps'])
            cosine = optim.cosine_decay(initial_lr, self.num_training_steps, initial_lr * cfg['min_lr_ratio'])
            return optim.join_schedules([warmup, cosine], [cfg['warmup_steps']])
        elif cfg['type'] == 'cosine':
            return optim.cosine_decay(initial_lr, self.num_training_steps, initial_lr * cfg['min_lr_ratio'])
        elif cfg['type'] == 'linear':
            return optim.linear_schedule(initial_lr, 0, steps=self.num_training_steps)
        else:
            raise ValueError(f"Unsupported scheduler type: {cfg['type']}")

    def create_optimizer(self, schedule: Any) -> optim.Optimizer:
        cfg = self.config.optimization
        kwargs = {
            'learning_rate': schedule,
        }
        if 'betas' in cfg:
            kwargs['betas'] = tuple(cfg['betas'])
        if 'eps' in cfg:
            kwargs['eps'] = cfg['eps']
        if 'weight_decay' in cfg:
            kwargs['weight_decay'] = self.config.hyperparameters['weight_decay']
        if cfg['optimizer'] == 'adamw':
            return optim.AdamW(**kwargs)
        elif cfg['optimizer'] == 'adam':
            return optim.Adam(**kwargs)
        elif cfg['optimizer'] == 'muon':
            return optim_x.Muon(**kwargs, alternate_optimizer=optim.AdamW(**kwargs))
        elif cfg['optimizer'] == 'sgd':
            return optim.SGD(**kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg['optimizer']}")

class Trainer:
    def __init__(self, config_path: str, for_training=True):
        self.config = Config.from_yaml(config_path)
        self.config_path = config_path

        # Initialize tracking variables
        self.total_tokens = 0
        self.start_step = 0

        # Validate unique run name before proceeding
        if for_training and not self.config.overwrite and not (self.config.resume and self.config.resume.checkpoint):
            CheckpointManager.validate_unique_name(self.config.name)

        self.setup_system()

        # Create run directory early so we can copy tokenizer to it
        if for_training:
            self.run_dir, self.log_file, self.checkpoint_dir = CheckpointManager.setup_run_directory(self.config.name)
        else:
            self.run_dir = None

        # Initialize tokenizer with run directory if available
        self.tokenizer = TokenizerManager(self.config.data, self.run_dir)

        self.setup_model()
        if for_training:
            self.data_manager = DataManager(self.config.data, self.tokenizer, batch_size=self.config.training.hyperparameters['batch_size'])
            self.setup_training()
            self.setup_logging()

            # Initialize validation metrics tracking
            self.validation_steps = self.config.logging.steps.get('validation_interval', 0)
            self.validation_losses = []

            # Initialize VLM evaluator if vision-language evaluation is enabled
            self.vlm_evaluator = None
            if self.config.data.is_vision_language and self.config.data.vlm_evaluation:
                self.vlm_evaluator = VLMEvaluator(self.config.data.vlm_evaluation, self.tokenizer)
                # Initialize dictionary to store validation metrics
                self.validation_metrics = []

    def setup_system(self):
        # Set random seeds
        random.seed(self.config.system.seed)
        np.random.seed(self.config.system.seed)
        mx.random.seed(self.config.system.seed)

    def setup_model(self):
        model_cfg = self.config.model
        arch_file = f"arch.{model_cfg.architecture}"
        mlx_lm_file = f"mlx_lm.models.{model_cfg.architecture}"
        mlx_vlm_file = f"mlx_vlm.models.{model_cfg.architecture}"
        Model = None
        ModelArgs = None
        is_vlm_model = False

        # Check if model_source is specified
        if model_cfg.model_source:
            if model_cfg.model_source == "custom":
                try:
                    module = importlib.import_module(arch_file)
                    Model = getattr(module, 'Model')
                    ModelArgs = getattr(module, 'ModelArgs')
                except ImportError:
                    raise ImportError(f"Custom model architecture '{model_cfg.architecture}' not found in {arch_file}")
            elif model_cfg.model_source == "mlx_lm":
                try:
                    module = importlib.import_module(mlx_lm_file)
                    Model = getattr(module, 'Model')
                    ModelArgs = getattr(module, 'ModelArgs')
                except ImportError:
                    raise ImportError(f"MLX-LM model architecture '{model_cfg.architecture}' not found in {mlx_lm_file}")
            elif model_cfg.model_source == "mlx_vlm":
                try:
                    module = importlib.import_module(mlx_vlm_file)
                    Model = getattr(module, 'Model')
                    ModelArgs = getattr(module, 'ModelArgs')
                    is_vlm_model = True
                except ImportError:
                    raise ImportError(f"MLX-VLM model architecture '{model_cfg.architecture}' not found in {mlx_vlm_file}")
            else:
                raise ValueError(f"Invalid model_source: {model_cfg.model_source}. Must be 'custom', 'mlx_lm', or 'mlx_vlm'")
        else:
            # Fallback to the original behavior if model_source is not specified
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
                    try:
                        module = importlib.import_module(mlx_vlm_file)
                        Model = getattr(module, 'Model')
                        ModelArgs = getattr(module, 'ModelArgs')
                        is_vlm_model = True
                    except ImportError:
                        raise ImportError(f"Model architecture '{model_cfg.architecture}' not found in {arch_file}, {mlx_lm_file}, or {mlx_vlm_file}")

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

        if self.config.data.weight_path is not None:
            print(f"Loading weights from {self.config.data.weight_path}")
            self.model.load_weights(self.config.data.weight_path, strict=False)

        # Apply LoRA/QLoRA if enabled for vision-language models
        if is_vlm_model and self.config.data.is_vision_language and model_cfg.lora is not None:
            try:
                # Import LoRA functions from mlx_vlm
                from mlx_vlm.lora import apply_lora_layers

                # Get LoRA configuration
                lora_cfg = model_cfg.lora
                lora_rank = lora_cfg.get('rank', 8)
                lora_alpha = lora_cfg.get('alpha', 16)
                lora_dropout = lora_cfg.get('dropout', 0.05)
                target_modules = lora_cfg.get('target_modules', ['q_proj', 'v_proj'])

                # Check if QLoRA is enabled
                use_qlora = lora_cfg.get('use_qlora', False)
                qlora_bits = lora_cfg.get('qlora_bits', 4)
                qlora_group_size = lora_cfg.get('qlora_group_size', 64)

                if use_qlora:
                    try:
                        # Try to import quantization functions
                        from mlx_vlm.lora import apply_qlora_layers

                        # Apply QLoRA to the model
                        print(f"Applying QLoRA with rank={lora_rank}, alpha={lora_alpha}, bits={qlora_bits}, group_size={qlora_group_size}")
                        self.model = apply_qlora_layers(
                            self.model,
                            rank=lora_rank,
                            alpha=lora_alpha,
                            dropout=lora_dropout,
                            target_modules=target_modules,
                            bits=qlora_bits,
                            group_size=qlora_group_size
                        )
                        print("QLoRA applied successfully")
                    except (ImportError, AttributeError) as e:
                        print(f"Warning: Could not apply QLoRA - {e}")
                        print("Falling back to standard LoRA")

                        # Fall back to standard LoRA
                        print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
                        self.model = apply_lora_layers(
                            self.model,
                            rank=lora_rank,
                            alpha=lora_alpha,
                            dropout=lora_dropout,
                            target_modules=target_modules
                        )
                        print("LoRA applied successfully")
                else:
                    # Apply standard LoRA to the model
                    print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
                    self.model = apply_lora_layers(
                        self.model,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        dropout=lora_dropout,
                        target_modules=target_modules
                    )
                    print("LoRA applied successfully")
            except ImportError as e:
                print(f"Warning: Could not apply LoRA/QLoRA - {e}")
                print("Make sure you have the latest version of mlx_vlm installed")
            except Exception as e:
                print(f"Error applying LoRA/QLoRA: {e}")

        # Log model size
        p = sum(v.size for _, v in tree_flatten(self.model.trainable_parameters())) / 10**6
        print(f"Model has {p:.2f}M parameters")

    def setup_training(self):
        # Calculate number of training steps
        num_samples = len(self.data_manager.train_docs)
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
        # Run directory structure should already be set up in __init__

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

        with open(self.run_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save the config used to the run directory
        with open(self.run_dir / 'config.yaml', 'w') as f:
            with open(self.config_path, 'r') as config_file:
                f.write(config_file.read())

    def compute_loss(self, model, inputs: Union[mx.array, Dict[str, Any]], targets: Union[mx.array, Dict[str, Any]]) -> Tuple[mx.array, int]:
        """Compute loss for the model.

        For text-only models, inputs and targets are mx.arrays of token IDs.
        For vision-language models, inputs is a dictionary with 'text_tokens' and 'images',
        and targets is the text_tokens shifted by one position.
        """
        if self.config.data.is_vision_language:
            # Vision-language model
            if not isinstance(inputs, dict):
                raise ValueError("Expected dictionary input for vision-language model")

            # Extract text tokens and images
            text_tokens = inputs["text_tokens"]
            images = inputs["images"]

            # Forward pass with both text and images
            model_output = model(text_tokens, images=images)

            # Check if the model returns a dictionary with multiple outputs
            if isinstance(model_output, dict):
                logits = model_output.get("logits", model_output.get("text_logits", None))
                image_embeddings = model_output.get("image_embeddings", None)
                text_embeddings = model_output.get("text_embeddings", None)
            else:
                # If model returns only logits (default behavior)
                logits = model_output
                image_embeddings = None
                text_embeddings = None

            # Ensure logits are float32 for numerical stability
            if logits is not None:
                logits = logits.astype(mx.float32)

            # Initialize total loss and components
            total_loss = 0.0
            loss_components = {}

            # Get multimodal loss configuration
            multimodal_loss_cfg = self.config.data.multimodal_loss or {}

            # 1. Language modeling loss (cross-entropy on text tokens)
            if logits is not None:
                lm_loss_weight = multimodal_loss_cfg.get("lm_loss_weight", 1.0)
                lm_loss = nn.losses.cross_entropy(logits, targets)

                # Mask padding tokens
                pad_mask = (targets != self.tokenizer.PAD_TOKEN)
                lm_loss = lm_loss * pad_mask
                ntoks = pad_mask.sum()

                # Normalize and weight the loss
                lm_loss = (lm_loss.sum() / ntoks) * lm_loss_weight
                total_loss += lm_loss
                loss_components["lm_loss"] = lm_loss
            else:
                ntoks = mx.array(1)  # Fallback if no language modeling is done

            # 2. Contrastive loss (if enabled and embeddings are available)
            if (multimodal_loss_cfg.get("enable_contrastive_loss", False) and 
                image_embeddings is not None and text_embeddings is not None):

                contrastive_loss_weight = multimodal_loss_cfg.get("contrastive_loss_weight", 0.5)
                temperature = multimodal_loss_cfg.get("contrastive_loss_temperature", 0.07)

                # Normalize embeddings
                image_embeddings = image_embeddings / (mx.linalg.norm(image_embeddings, axis=-1, keepdims=True) + 1e-8)
                text_embeddings = text_embeddings / (mx.linalg.norm(text_embeddings, axis=-1, keepdims=True) + 1e-8)

                # Compute similarity matrix
                similarity = mx.matmul(image_embeddings, text_embeddings.transpose())
                similarity = similarity / temperature

                # Labels are the diagonal elements (matching pairs)
                labels = mx.eye(similarity.shape[0])

                # Compute contrastive loss (cross-entropy with diagonal as targets)
                contrastive_loss_i2t = nn.losses.cross_entropy(similarity, labels)
                contrastive_loss_t2i = nn.losses.cross_entropy(similarity.transpose(), labels)
                contrastive_loss = (contrastive_loss_i2t + contrastive_loss_t2i) / 2.0

                # Weight the contrastive loss
                contrastive_loss = contrastive_loss * contrastive_loss_weight
                total_loss += contrastive_loss
                loss_components["contrastive_loss"] = contrastive_loss

            # 3. Image-text matching loss (if enabled and model provides matching scores)
            if multimodal_loss_cfg.get("enable_matching_loss", False):
                matching_loss_weight = multimodal_loss_cfg.get("matching_loss_weight", 0.5)

                # Check if model provides matching scores
                if isinstance(model_output, dict) and "matching_scores" in model_output:
                    matching_scores = model_output["matching_scores"]
                    matching_labels = mx.eye(matching_scores.shape[0])  # Diagonal is 1 (match), rest is 0 (no match)

                    # Binary cross-entropy loss for matching
                    matching_loss = nn.losses.binary_cross_entropy(
                        mx.sigmoid(matching_scores), 
                        matching_labels
                    )

                    # Weight the matching loss
                    matching_loss = matching_loss * matching_loss_weight
                    total_loss += matching_loss
                    loss_components["matching_loss"] = matching_loss

            # If no multimodal losses were added, fall back to standard cross-entropy
            if total_loss == 0.0 and logits is not None:
                total_loss = lm_loss

            # Log loss components if verbose logging is enabled
            if multimodal_loss_cfg.get("verbose_loss_logging", False) and len(loss_components) > 1:
                components_str = ", ".join([f"{k}={float(v):.4f}" for k, v in loss_components.items()])
                print(f"Loss components: {components_str}, total={float(total_loss):.4f}")

            return total_loss, ntoks
        else:
            # Text-only model (original behavior)
            logits = model(inputs)
            logits = logits.astype(mx.float32)
            loss = nn.losses.cross_entropy(logits, targets)

            # Mask padding tokens
            pad_mask = (targets != self.tokenizer.PAD_TOKEN)
            loss = loss * pad_mask
            ntoks = pad_mask.sum()

            return loss.sum() / ntoks, ntoks

    def validate(self) -> Union[float, Tuple[float, Dict[str, float]]]:
        """Run validation on the validation dataset.

        Returns:
            If vision-language evaluation is enabled:
                Tuple[float, Dict[str, float]]: Average validation loss and dictionary of validation metrics
            Otherwise:
                float: Average validation loss
        """
        if not self.data_manager.has_validation_data:
            return None

        # Ensure we're in evaluation mode (no need for gradients)
        total_loss = 0.0
        total_tokens = 0
        batch_metrics = []

        # Process all validation batches
        num_batches = min(self.data_manager.num_validation_batches, 50)  # Cap at 50 batches to avoid too long validation

        for batch_idx in range(num_batches):
            batch = self.data_manager.generate_validation_batch(batch_idx)

            # Forward pass only - handle different batch formats
            if self.config.data.is_vision_language:
                # Vision-language model
                text_tokens = batch["text_tokens"]
                inputs = {"text_tokens": text_tokens[:, :-1], "images": batch["images"]}
                targets = text_tokens[:, 1:]

                # Forward pass to get model output
                model_output = self.model(inputs["text_tokens"], images=inputs["images"])

                # Compute loss
                loss, tokens = self.compute_loss(self.model, inputs, targets)

                # Compute additional metrics if VLM evaluator is available
                if self.vlm_evaluator is not None:
                    metrics = self.vlm_evaluator.compute_metrics(self.model, batch, model_output)
                    batch_metrics.append(metrics)
            else:
                # Text-only model
                loss, tokens = self.compute_loss(self.model, batch[:, :-1], batch[:, 1:])

            # Accumulate loss
            total_loss += float(loss)
            total_tokens += tokens

            # Clear GPU cache if needed
            if self.config.system.device == "gpu":
                mx.clear_cache()

        # Calculate average loss
        avg_loss = total_loss / num_batches

        # Aggregate metrics if VLM evaluator is available
        if self.vlm_evaluator is not None and batch_metrics:
            aggregated_metrics = self.vlm_evaluator.aggregate_metrics(batch_metrics)

            # Log metrics if verbose logging is enabled
            if self.vlm_evaluator.verbose_metrics_logging:
                metrics_str = self.vlm_evaluator.format_metrics(aggregated_metrics)
                print(f"Validation metrics: {metrics_str}")

            return avg_loss, aggregated_metrics
        else:
            return avg_loss

    def save_checkpoint(self, step: int | str, val_result: Union[float, Tuple[float, Dict[str, float]]] = None):
        # Save model weights
        weights = dict(tree_flatten(self.model.parameters()))
        model_path = self.checkpoint_dir / f'step_{step}_model.safetensors'
        mx.save_safetensors(str(model_path), weights)

        # Save optimizer state
        optimizer_state = dict(tree_flatten(self.optimizer.state))
        optimizer_path = self.checkpoint_dir / f'step_{step}_optimizer.safetensors'
        mx.save_safetensors(str(optimizer_path), optimizer_state)

        # Extract validation loss and metrics
        val_loss = None
        val_metrics = None
        if val_result is not None:
            if isinstance(val_result, tuple) and len(val_result) == 2:
                val_loss, val_metrics = val_result
            else:
                val_loss = val_result

        # Save training state
        training_state = {
            'step': step if isinstance(step, int) else self.total_steps,
            'val_ptr': self.data_manager.val_ptr,
            'total_tokens': self.total_tokens.item(),
            'validation_losses': self.validation_losses,
        }

        # Include validation metrics if available
        if hasattr(self, 'validation_metrics') and self.validation_metrics:
            training_state['validation_metrics'] = self.validation_metrics

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

        # Include validation metrics if available
        if val_metrics is not None:
            checkpoint_info['validation_metrics'] = val_metrics

        metadata['checkpoints'].append(checkpoint_info)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def log_metrics(self, step: int, loss: float, tokens: int, 
                   total_tokens: int, start_time: float, val_result: Union[float, Tuple[float, Dict[str, float]]] = None) -> str:
        metrics = []

        # Extract validation loss and metrics
        val_loss = None
        val_metrics = None
        if val_result is not None:
            if isinstance(val_result, tuple) and len(val_result) == 2:
                val_loss, val_metrics = val_result
            else:
                val_loss = val_result

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

            # Add validation perplexity if available
            if val_loss is not None:
                metrics.append(f"val_ppl={np.exp(val_loss):.2f}")

        if self.config.logging.metrics['log_tokens_per_second']:
            tokens_per_sec = total_tokens / (1000 * (time.time() - start_time))
            metrics.append(f"tok/s={tokens_per_sec:.2f}K")

        if self.config.logging.metrics['log_tokens_processed']:
            metrics.append(f"toks={tokens}")

        if self.config.logging.metrics['log_learning_rate']:
            metrics.append(f"lr={self.lr_schedule(step):.3e}")

        # Add vision-language metrics if available
        if val_metrics is not None and self.vlm_evaluator is not None:
            # Format key metrics for the log output (limit to a few important ones)
            vlm_metrics = []

            # Add retrieval metrics if available
            for k in self.vlm_evaluator.retrieval_k_values:
                if f"mean_recall@{k}" in val_metrics:
                    vlm_metrics.append(f"R@{k}={val_metrics[f'mean_recall@{k}']:.4f}")

            # Add other important metrics (limited to keep the log output readable)
            important_metrics = ["vqa_accuracy", "classification_accuracy"]
            for key in important_metrics:
                if key in val_metrics:
                    vlm_metrics.append(f"{key}={val_metrics[key]:.4f}")

            # Add the formatted metrics to the log output
            if vlm_metrics:
                metrics.append("vlm_metrics: " + ", ".join(vlm_metrics))

        return " | ".join(metrics)

    def load_checkpoint(self, checkpoint_path: str, reset_optimizer: bool = False):
        """Load a checkpoint and restore model, optimizer, and training state"""
        # Extract step from checkpoint path
        step_str = checkpoint_path.split('step_')[-1]

        # Get checkpoint file paths
        model_path, optimizer_path, state_path = CheckpointManager.get_checkpoint_paths(checkpoint_path)

        # Load model weights
        print(f"Loading model weights from {model_path}")
        #weights = mx.load(model_path)
        self.model.load_weights(model_path)
        # Load optimizer state if not resetting
        if not reset_optimizer:
            print(f"Loading optimizer state from {optimizer_path}")
            state_dict = mx.load(optimizer_path)
            state = tree_unflatten(list(state_dict.items()))
            self.optimizer.state = state

        # Load training state
        print(f"Loading training state from {state_path}")
        with open(state_path, 'r') as f:
            training_state = json.load(f)

        # Restore training state
        self.start_step = training_state['step'] if isinstance(training_state['step'], int) else 0
        self.data_manager.val_ptr = training_state['val_ptr']
        self.total_tokens = training_state['total_tokens']
        self.validation_losses = training_state['validation_losses']

        print(f"Resumed training from checkpoint {checkpoint_path} at step {self.start_step}")

        return self.start_step

    def train(self):
        # Initialize variables
        total_tokens = self.total_tokens
        start_step = 0

        # Check if resuming from checkpoint
        if self.config.resume and self.config.resume.checkpoint:
            checkpoint_path = self.config.resume.checkpoint
            reset_optimizer = self.config.resume.reset_optimizer
            start_step = self.load_checkpoint(checkpoint_path, reset_optimizer)

            # If we're resuming, we should skip the initial validation
            skip_initial_validation = True
        else:
            skip_initial_validation = False

        loss_value_and_grad = nn.value_and_grad(self.model, self.compute_loss)
        start_time = time.time()
        # Create progress bar with adjusted range for resuming
        progress_bar = tqdm(range(self.total_steps), desc="Training", initial=start_step)


        # Initialize logging
        with open(self.log_file, 'a' if start_step > 0 else 'w') as log_file:
            if start_step == 0:
                log_file.write(f"Training started at {datetime.now()}\n")
                log_file.write(f"Total steps: {self.total_steps}\n")
                if self.config.training.epochs is not None:
                    log_file.write(f"Training for {self.config.training.epochs} epochs with {self.steps_per_epoch} steps per epoch\n")
                if self.data_manager.has_validation_data:
                    log_file.write(f"Validation data: {self.config.data.validation_file}\n")
                    log_file.write(f"Validation batches: {self.data_manager.num_validation_batches}\n")
                log_file.write("=" * 50 + "\n\n")
            else:
                log_file.write(f"\nResuming training at step {start_step} at {datetime.now()}\n")
                log_file.write(f"Remaining steps: {self.total_steps - start_step}\n")
                log_file.write("=" * 50 + "\n\n")

            # Log initial validation loss if validation data is available and not resuming
            val_result = None
            if self.validation_steps > 0 and self.data_manager.has_validation_data and not skip_initial_validation:
                val_result = self.validate()

                # Extract validation loss and metrics
                if isinstance(val_result, tuple) and len(val_result) == 2:
                    val_loss, val_metrics = val_result
                    # Add to validation metrics history
                    if hasattr(self, 'validation_metrics'):
                        self.validation_metrics.append((0, val_metrics))
                    # Log validation metrics
                    if val_metrics and self.vlm_evaluator and self.vlm_evaluator.verbose_metrics_logging:
                        metrics_str = self.vlm_evaluator.format_metrics(val_metrics)
                        log_file.write(f"Initial validation metrics: {metrics_str}\n")
                else:
                    val_loss = val_result

                log_file.write(f"Initial validation loss: {val_loss:.4e} (ppl={np.exp(val_loss):.2f})\n\n")
                # Add to validation loss history
                self.validation_losses.append((0, val_loss))

            for step in progress_bar:
                step += start_step
                if step >= self.total_steps:
                    break
                # Generate batch
                batch = self.data_manager.generate_batch(step)

                # Forward and backward pass - handle different batch formats
                if self.config.data.is_vision_language:
                    # Vision-language model
                    text_tokens = batch["text_tokens"]
                    # Compute loss with text_tokens shifted by one position
                    (loss, tokens), grad = loss_value_and_grad(
                        self.model, 
                        {"text_tokens": text_tokens[:, :-1], "images": batch["images"]}, 
                        text_tokens[:, 1:]
                    )
                else:
                    # Text-only model
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
                if self.validation_steps > 0 and self.data_manager.has_validation_data and (step + 1) % self.validation_steps == 0:
                    val_result = self.validate()

                    # Extract validation loss and metrics
                    if isinstance(val_result, tuple) and len(val_result) == 2:
                        val_loss, val_metrics = val_result
                        # Add to validation metrics history
                        if hasattr(self, 'validation_metrics'):
                            self.validation_metrics.append((step + 1, val_metrics))

                        # Log validation metrics separately for clear visibility
                        if val_metrics and self.vlm_evaluator:
                            metrics_str = self.vlm_evaluator.format_metrics(val_metrics)
                            log_file.write(f"Step {step + 1} validation metrics: {metrics_str}\n")
                    else:
                        val_loss = val_result

                    # Add to validation loss history
                    self.validation_losses.append((step + 1, val_loss))

                    # Log validation loss separately for clear visibility
                    val_loss_str = f"val_loss={val_loss:.3e} | val_ppl={np.exp(val_loss):.2f}"
                    log_file.write(f"Step {step + 1} validation: {val_loss_str}\n")
                    log_file.flush()

                # Logging
                if step % self.config.logging.steps['logging_interval'] == 0:
                    # Only include val_loss if it was just calculated
                    current_val_loss = val_loss if self.validation_steps > 0 and (step + 1) % self.validation_steps == 0 else None
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
        final_val_result = None
        if self.validation_steps > 0 and self.data_manager.has_validation_data:
            final_val_result = self.validate()

            # Extract validation loss and metrics
            if isinstance(final_val_result, tuple) and len(final_val_result) == 2:
                final_val_loss, final_val_metrics = final_val_result
                # Add to validation metrics history
                if hasattr(self, 'validation_metrics'):
                    self.validation_metrics.append((self.total_steps, final_val_metrics))
            else:
                final_val_loss = final_val_result

            # Add to validation loss history
            self.validation_losses.append((self.total_steps, final_val_loss))

        # Save final checkpoint with validation result
        self.total_tokens = total_tokens
        self.save_checkpoint("final", final_val_result)

        # Save validation losses and metrics to metadata
        if self.validation_losses:
            metadata_path = self.run_dir / 'metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            metadata['validation'] = {
                'steps': [step for step, _ in self.validation_losses],
                'losses': [float(loss) for _, loss in self.validation_losses]
            }

            # Save validation metrics if available
            if hasattr(self, 'validation_metrics') and self.validation_metrics:
                # Convert metrics to serializable format
                serializable_metrics = []
                for step, metrics in self.validation_metrics:
                    serializable_metrics.append({
                        'step': step,
                        'metrics': {k: float(v) for k, v in metrics.items()}
                    })

                metadata['validation']['metrics'] = serializable_metrics

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        # Write final summary
        with open(self.log_file, 'a') as log_file:
            log_file.write("\n" + "=" * 50 + "\n")
            log_file.write(f"Training completed at {datetime.now()}\n")
            log_file.write(f"Final training metrics: {metrics}\n")

            # Log final validation results
            if 'final_val_loss' in locals() and final_val_loss is not None:
                log_file.write(f"Final validation loss: {final_val_loss:.4e} (ppl={np.exp(final_val_loss):.2f})\n")

                # Log final validation metrics if available
                if 'final_val_metrics' in locals() and final_val_metrics is not None and self.vlm_evaluator is not None:
                    metrics_str = self.vlm_evaluator.format_metrics(final_val_metrics)
                    log_file.write(f"Final validation metrics: {metrics_str}\n")

            log_file.write(f"Total tokens processed: {total_tokens/1000:.2f}K\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train a language model with MLX')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    args = parser.parse_args()
    # Make 'runs' directory if it doesn't exist
    os.makedirs('runs', exist_ok=True)
    trainer = Trainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main()
