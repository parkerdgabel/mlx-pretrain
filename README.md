# MLX-Pretrain

MLX-Pretrain is a library for training and fine-tuning language models using Apple's MLX framework.

## Table of Contents

- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft-with-mlx-pretrain)
- [Reinforcement Learning (RL)](#reinforcement-learning-with-mlx-pretrain)

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