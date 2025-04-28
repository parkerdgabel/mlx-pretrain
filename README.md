# MLX-Pretrain

MLX-Pretrain is a library for training and fine-tuning language models using Apple's MLX framework.

## Table of Contents

- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft-with-mlx-pretrain)
- [Reinforcement Learning (RL)](#reinforcement-learning-with-mlx-pretrain)
- [Vision-Language Models](#vision-language-models-with-mlx-pretrain)

# Supervised Fine-Tuning (SFT) with MLX-Pretrain

This section explains how to use the `sft.py` script for supervised fine-tuning of language models with MLX-Pretrain.

## Overview

The `sft.py` script allows you to fine-tune pretrained language models on specific tasks using supervised learning. This process adapts a general-purpose language model to perform better on targeted tasks by training it on examples of desired inputs and outputs.

## Prerequisites

Before using the SFT script, you should:

1. Have a pretrained model (from MLX-Pretrain or another compatible source)
2. Prepare your training data in the required format
3. Create a configuration file for the fine-tuning process

## Data Preparation

The SFT script expects data in JSONL (JSON Lines) format, where each line contains a separate JSON object with instruction-response pairs:

```json
{"instruction": "Write a poem about machine learning", "response": "Silicon minds in constant flow,\nPatterns emerge as data grows..."}
```

You can customize the field names in the configuration file if your data uses different keys.

### Example Data Format

```json
{"instruction": "Explain how photosynthesis works", "response": "Photosynthesis is the process by which plants..."}
{"instruction": "What are the key differences between Python and JavaScript?", "response": "Python and JavaScript differ in several key ways..."}
```

## Configuration

Create a YAML configuration file to specify the model architecture, training parameters, and data formatting. Here's an example:

```yaml
name: "Llama-2M-SFT"
overwrite: true
data:
  input_file: "sft_data.jsonl"
  validation_file: "sft_val.jsonl"
  tokenizer_path: "runs/Llama (2M)/tokenizer"

  # SFT specific configuration
  prompt_format: "### Instruction:\n{instruction}\n\n### Response:"
  response_format: "{response}"
  system_prompt: "You are a helpful, harmless, and honest AI assistant."
  input_field: "instruction"
  output_field: "response"

  preprocessing:
    max_context_size: 1024

model:
  architecture: "llama"
  dimensions:
    hidden_size: 2048
    intermediate_size: 4096
    num_layers: 12
  # Additional model parameters...

training:
  epochs: 3
  hyperparameters:
    batch_size: 8
    learning_rate: 5.0e-5
    weight_decay: 0.01
    gradient_clip: 1.0
  # Additional training parameters...
```

### Key Configuration Sections

- **data**: Specifies input files, tokenizer, and formatting templates
- **model**: Defines the model architecture and dimensions
- **training**: Sets training hyperparameters, optimizer, and scheduler
- **logging**: Controls checkpointing and metric logging
- **system**: Sets system-level parameters like random seed and device

## Usage

Run the SFT script with the following command:

```bash
python sft.py --config sft-config.yaml --pretrained_model "path/to/pretrained/model"
```

### Arguments

- `--config`: (Required) Path to the SFT configuration YAML file
- `--pretrained_model`: (Optional) Path to a pretrained model directory or checkpoint

## Output and Checkpoints

The script creates a run directory with:

- Checkpoints saved at intervals specified in the config
- Training logs with metrics
- Metadata about the training run

## Customizing Prompts and Responses

You can customize how instructions and responses are formatted using the `prompt_format` and `response_format` settings:

```yaml
prompt_format: "### Instruction:\n{instruction}\n\n### Response:"
response_format: "{response}"
system_prompt: "You are a helpful, harmless, and honest AI assistant."
```

This allows you to match the format used during pretraining or to prepare the model for specific deployment scenarios.

## Tips for Effective Fine-Tuning

1. **Start with a well-pretrained model**: The quality of your base model significantly impacts fine-tuning results.
2. **Use a lower learning rate**: Fine-tuning typically requires lower learning rates (1e-5 to 5e-5) than pretraining.
3. **Enable gradient clipping**: This helps prevent training instability, especially with smaller batch sizes.
4. **Monitor validation loss**: Use the validation_interval parameter to regularly check performance on held-out data.
5. **Experiment with prompt formats**: The way you format instructions can significantly impact model performance.

## Example Workflow

1. Prepare your SFT data in JSONL format
2. Create or modify an SFT configuration file
3. Run the SFT script with your configuration and pretrained model
4. Monitor training progress through the logs
5. Evaluate the fine-tuned model on your target tasks

## Advanced Usage

For more advanced use cases, you can:

- Customize the optimizer and learning rate scheduler
- Implement custom data preprocessing
- Adjust the tokenization process
- Fine-tune specific layers while freezing others

Refer to the code documentation for more details on these advanced features.

# Reinforcement Learning with MLX-Pretrain

This section explains how to use the reinforcement learning (RL) functionality in MLX-Pretrain.

## Overview

The RL script (`rl.py`) allows you to fine-tune pretrained models using reinforcement learning from human feedback (RLHF) techniques, specifically Proximal Policy Optimization (PPO). This approach can help align language models with human preferences.

## Prerequisites

Before using the RL script, you should:

1. Have a pretrained model (either from pretraining or SFT)
2. Prepare RL data with prompts, responses, and reward values
3. Optionally, have a reward model (or use the default reward implementation)

## Preparing RL Data

Create JSONL files with the following format:

```json
{
  "instruction": "Write a poem about machine learning",
  "response": "Machines learning day by day...",
  "reward": 0.8
}
```

You can customize the field names in the configuration file.

## Configuration

Use the `rl-config.yaml` file to configure your RL training. Key parameters include:

- **PPO-specific parameters**: `ppo_epochs`, `ppo_mini_batch_size`, `kl_coef`, `clip_range`
- **Data configuration**: Input/output formats, field names, etc.
- **Training hyperparameters**: Learning rate, batch size, etc.

Example:

```yaml
data:
  # RL specific configuration
  prompt_format: "### Instruction:\n{instruction}\n\n### Response:"
  response_format: "{response}"
  reward_field: "reward"

  # PPO specific parameters
  ppo_epochs: 4
  ppo_mini_batch_size: 8
  kl_coef: 0.1
  clip_range: 0.2
```

## Running RL Training

To start RL training:

```bash
python rl.py --config rl-config.yaml --pretrained_model "runs/Llama-2M-SFT"
```

If you have a custom reward model:

```bash
python rl.py --config rl-config.yaml --pretrained_model "runs/Llama-2M-SFT" --reward_model "path/to/reward_model"
```

## Customizing the Reward Model

The default `RewardModel` class can be extended to implement custom reward functions:

```python
class CustomRewardModel(RewardModel):
    def compute_reward(self, prompts, responses):
        # Implement your custom reward logic here
        return rewards
```

## Monitoring Training

During training, the script logs various metrics:

- Policy loss
- KL divergence
- Rewards
- Training/validation loss
- Tokens per second

These metrics are saved to the run directory and can be visualized using the plotting tools.

## Tips for Effective RL Training

1. **Start with a well-trained SFT model**: RL works best when starting from a model that already performs reasonably well.
2. **Tune the KL coefficient**: The `kl_coef` parameter controls how much the model can deviate from the reference model. Higher values result in more conservative updates.
3. **Use appropriate learning rates**: RL typically requires lower learning rates than SFT (around 1e-5 to 5e-5).
4. **Monitor rewards and KL divergence**: These metrics help you understand if your model is improving while staying close to the reference policy.
5. **Ensure quality reward signals**: The quality of your reward model or reward function is critical for successful RL training.

## Example Workflow

1. Pretrain a base model: `python train.py --config model-config.yaml`
2. Fine-tune with SFT: `python sft.py --config sft-config.yaml --pretrained_model "runs/Llama (2M)"`
3. Fine-tune with RL: `python rl.py --config rl-config.yaml --pretrained_model "runs/Llama-2M-SFT"`

# Vision-Language Models with MLX-Pretrain

This section explains how to train and fine-tune vision-language models (VLMs) using MLX-Pretrain.

## Overview

Vision-Language Models combine visual and textual understanding, allowing models to process both images and text. MLX-Pretrain now supports training and fine-tuning of VLMs with sophisticated multimodal loss functions.

## Prerequisites

Before training a vision-language model, you should:

1. Have a pretrained vision-language model (from mlx_vlm or another compatible source)
2. Prepare your training data with both text and image paths
3. Create a configuration file that enables vision-language features

## Data Preparation

For vision-language models, your JSONL data should include both text and image paths:

```json
{"text": "A photo of a cat sitting on a windowsill", "images": ["path/to/cat_image.jpg"]}
{"text": "Two dogs playing in the park", "images": ["path/to/dogs_image.jpg"]}
```

For multi-image examples:

```json
{"text": "Compare these two images", "images": ["path/to/image1.jpg", "path/to/image2.jpg"]}
```

## Configuration

Create a YAML configuration file with vision-language specific settings:

```yaml
name: "VLM-Training"
overwrite: true
data:
  input_file: "vlm_data.jsonl"
  validation_file: "vlm_val.jsonl"
  tokenizer_path: "path/to/tokenizer"

  # Vision-language specific configuration
  is_vision_language: true
  image_field: "images"
  text_field: "text"
  image_processor_path: "path/to/image/processor"  # Optional
  max_images_per_sample: 5  # Maximum number of images per sample

  # Multimodal loss configuration
  multimodal_loss:
    lm_loss_weight: 1.0                # Weight for language modeling loss
    enable_contrastive_loss: true      # Enable contrastive loss
    contrastive_loss_weight: 0.5       # Weight for contrastive loss
    contrastive_loss_temperature: 0.07 # Temperature for contrastive loss
    enable_matching_loss: true         # Enable image-text matching loss
    matching_loss_weight: 0.3          # Weight for matching loss
    verbose_loss_logging: true         # Enable verbose logging of loss components

  preprocessing:
    max_context_size: 1024

model:
  architecture: "qwen2_vl"  # Vision-language model architecture
  dimensions:
    hidden_size: 2048
    intermediate_size: 4096
    num_layers: 12
  # Additional model parameters...

  # LoRA/QLoRA configuration for efficient fine-tuning
  lora:
    rank: 8                      # Rank of the LoRA matrices
    alpha: 16                    # Scaling factor for LoRA matrices
    dropout: 0.05                # Dropout rate for LoRA layers
    target_modules: ["q_proj", "v_proj"]  # Modules to apply LoRA to

    # QLoRA specific parameters (optional)
    use_qlora: true              # Enable QLoRA (quantized LoRA)
    qlora_bits: 4                # Number of bits for quantization (4 or 8)
    qlora_group_size: 64         # Group size for quantization

training:
  epochs: 3
  hyperparameters:
    batch_size: 4  # Smaller batch size due to memory requirements
    learning_rate: 2.0e-5
    weight_decay: 0.01
    gradient_clip: 1.0
  # Additional training parameters...
```

## Multimodal Loss Functions

MLX-Pretrain supports several loss functions specifically designed for vision-language models:

1. **Language Modeling Loss**: Standard cross-entropy loss on text tokens (always enabled)
2. **Contrastive Loss**: Aligns image and text representations in the embedding space
3. **Image-Text Matching Loss**: Helps the model determine if an image and text pair match

You can configure these loss functions using the `multimodal_loss` section in your configuration file.

## Evaluation Metrics for Vision-Language Models

MLX-Pretrain now supports comprehensive evaluation metrics for vision-language models:

### Available Metrics

1. **Image-Text Retrieval Metrics**:
   - **Recall@K**: Measures the percentage of times the correct image/text is retrieved within the top K results
   - **Mean Recall**: Average of image-to-text and text-to-image retrieval performance

2. **Visual Question Answering Metrics** (placeholder implementation):
   - **VQA Accuracy**: Measures the accuracy of answers to visual questions

3. **Image Captioning Metrics** (placeholder implementation):
   - **BLEU Score**: Measures the quality of generated captions

4. **Zero-Shot Classification Metrics** (placeholder implementation):
   - **Classification Accuracy**: Measures the accuracy of zero-shot classification

### Configuration

Enable and configure evaluation metrics in your YAML configuration file:

```yaml
data:
  # Other data configuration...

  # Vision-language evaluation metrics configuration
  vlm_evaluation:
    enable_retrieval_metrics: true      # Enable image-text retrieval metrics
    retrieval_k_values: [1, 5, 10]      # K values for Recall@K metrics
    enable_vqa_metrics: false           # Enable visual question answering metrics
    enable_captioning_metrics: false    # Enable image captioning metrics
    enable_classification_metrics: false # Enable zero-shot classification metrics
    classification_categories: []        # Categories for zero-shot classification
    verbose_metrics_logging: true       # Enable verbose logging of metrics
```

### Interpreting Results

During training, validation metrics will be logged alongside the standard loss metrics:

```
Step 1000 validation: val_loss=2.345e-01 | val_ppl=1.26
Step 1000 validation metrics: i2t_recall@1=0.4500, t2i_recall@1=0.4200, mean_recall@1=0.4350, i2t_recall@5=0.7800, t2i_recall@5=0.7500, mean_recall@5=0.7650
```

The metrics are also saved in the run's metadata and can be visualized or analyzed after training.

## Running Vision-Language Training

To start training a vision-language model:

```bash
python train.py --config vlm-config.yaml
```

For fine-tuning:

```bash
python sft.py --config vlm-sft-config.yaml --pretrained_model "path/to/pretrained/vlm"
```

## Model Compatibility

MLX-Pretrain supports vision-language models from:

1. Custom architectures in the `arch` directory
2. Models from the `mlx_lm` package
3. Models from the `mlx_vlm` package

### Model Source Selection

MLX-Pretrain now supports explicitly selecting the source of your model architecture:

- **custom**: Models defined in the `arch` directory
- **mlx_lm**: Models from the `mlx_lm` package
- **mlx_vlm**: Vision-language models from the `mlx_vlm` package

You can specify the model source in your configuration file:

```yaml
model:
  architecture: "llama"
  model_source: "mlx_lm"  # Use "custom", "mlx_lm", or "mlx_vlm"
  dimensions:
    hidden_size: 2048
    # ...
```

If `model_source` is not specified, MLX-Pretrain will try to load the model from each source in the following order:
1. Custom models from the `arch` directory
2. Models from `mlx_lm`
3. Models from `mlx_vlm`

## Tips for Effective Vision-Language Training

1. **Use appropriate batch sizes**: Vision-language models require more memory, so you may need to reduce batch size.
2. **Balance loss components**: Adjust the weights of different loss components to achieve the best performance.
3. **Use LoRA or QLoRA for efficient fine-tuning**: 
   - LoRA allows efficient adaptation of large vision-language models with fewer parameters.
   - QLoRA (Quantized LoRA) further reduces memory requirements by quantizing the base model weights.
   - For very large models, enable QLoRA with `use_qlora: true` in your configuration.
4. **Monitor individual loss components**: Use `verbose_loss_logging: true` to track how each loss component evolves.
5. **Ensure image diversity**: Include a variety of images to help the model generalize across different visual concepts.
6. **Consider memory-performance tradeoffs**: 
   - Adjust QLoRA parameters (`qlora_bits` and `qlora_group_size`) to balance memory usage and model quality.
   - Lower bit precision (4-bit) saves more memory but may slightly impact performance.

## Example Workflow

1. Prepare vision-language data with text and image paths
2. Create a configuration file with vision-language settings
3. Train or fine-tune the model using the appropriate script
4. Monitor the individual loss components during training
5. Evaluate the model on vision-language tasks
