import pytest
import os
import tempfile
import yaml
import mlx.core as mx
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from rl import (
    RLConfig, RLDataConfig, RLDataManager, RewardModel, RLTrainer
)

# Test RLConfig loading
def test_rl_config_from_yaml():
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+') as f:
        yaml.dump({
            'data': {
                'input_file': 'test.jsonl',
                'prompt_format': '{prompt}',
                'reward_model': 'reward_model_path'
            },
            'model': {'architecture': 'llama'},
            'training': {'batch_size': 4},
            'logging': {'log_interval': 10},
            'system': {'seed': 42},
        }, f)
        f.flush()
        
        # Test loading
        config = RLConfig.from_yaml(f.name)
        
        # Assertions
        assert isinstance(config.data, RLDataConfig)
        assert config.data.input_file == 'test.jsonl'
        assert config.data.prompt_format == '{prompt}'
        assert config.data.reward_model == 'reward_model_path'
        assert config.model.architecture == 'llama'
        assert config.training.batch_size == 4

# Test RLDataManager
def test_rl_data_manager_batch_generation():
    # Mock dependencies
    tokenizer = MagicMock()
    tokenizer.tokenize.return_value = [1, 2, 3, 4, 5]
    tokenizer.VOCAB_SIZE = 100
    
    config = MagicMock()
    config.input_file = 'test.jsonl'
    config.validation_file = None
    config.prompt_format = '{prompt}'
    
    # Mock data loading
    with patch('rl.RLDataManager._load_file'):
        with patch('rl.RLDataManager._create_batch', return_value={'prompts': mx.array([1, 2, 3]), 'prompt_lengths': mx.array([3])}):
            data_manager = RLDataManager(config, tokenizer, batch_size=2)
            data_manager.data = [{'prompt': 'test prompt'}] * 10
            
            # Test batch generation
            batch = data_manager.generate_batch(0)
            assert 'prompts' in batch
            assert 'prompt_lengths' in batch

# Test RewardModel
def test_reward_model():
    # Mock model loading
    with patch('rl.RewardModel._load_model'):
        reward_model = RewardModel(model_path='reward_model_path')
        
        # Mock compute_reward
        reward_model.model = MagicMock()
        reward_model.model.return_value = {'rewards': mx.array([0.5, 0.7])}
        
        # Test reward computation
        prompts = mx.array([[1, 2], [3, 4]])
        responses = mx.array([[5, 6], [7, 8]])
        
        rewards = reward_model.compute_reward(prompts, responses)
        assert isinstance(rewards, mx.array)
        assert rewards.shape == (2,)

# Test RLTrainer
@pytest.mark.parametrize("pretrained_model_path,reward_model_path", [
    (None, None),
    ("pretrained_model", "reward_model")
])
def test_rl_trainer_initialization(pretrained_model_path, reward_model_path):
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+') as f:
        yaml.dump({
            'data': {
                'input_file': 'test.jsonl',
                'prompt_format': '{prompt}',
                'reward_model': 'reward_model_path',
                'tokenizer': {
                    'path': 'tokenizer',
                    'special_tokens': {
                        'bos': '<s>',
                        'eos': '</s>',
                        'pad': '<pad>',
                        'unk': '<unk>'
                    }
                }
            },
            'model': {
                'architecture': 'llama',
                'dimensions': {
                    'hidden_size': 128,
                    'intermediate_size': 256,
                    'num_layers': 2
                },
                'attention': {'num_heads': 4}
            },
            'training': {
                'batch_size': 4,
                'optimizer': 'adamw',
                'learning_rate': 1e-4
            },
            'logging': {'log_interval': 10},
            'system': {'seed': 42},
        }, f)
        f.flush()
        
        # Mock dependencies
        with patch('rl.RLTrainer.setup_model'):
            with patch('rl.RLTrainer.setup_training'):
                with patch('rl.RLTrainer.setup_logging'):
                    with patch('train.TokenizerManager'):
                        with patch('os.makedirs'):
                            with patch('rl.RewardModel'):
                                # Test initialization
                                trainer = RLTrainer(
                                    f.name, 
                                    pretrained_model_path=pretrained_model_path,
                                    reward_model_path=reward_model_path
                                )
                                
                                # Assertions
                                assert trainer.config is not None
                                assert trainer.pretrained_model_path == pretrained_model_path
                                assert trainer.reward_model_path == reward_model_path