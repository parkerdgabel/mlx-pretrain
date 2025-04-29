import pytest
import os
import tempfile
import yaml
import mlx.core as mx
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from sft import (
    SFTConfig, SFTDataConfig, SFTDataManager, SFTTrainer
)

# Test SFTConfig loading
def test_sft_config_from_yaml():
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+') as f:
        yaml.dump({
            'data': {
                'input_file': 'test.jsonl',
                'prompt_format': '{prompt}',
                'response_format': '{response}'
            },
            'model': {'architecture': 'llama'},
            'training': {'batch_size': 4},
            'logging': {'log_interval': 10},
            'system': {'seed': 42},
        }, f)
        f.flush()
        
        # Test loading
        config = SFTConfig.from_yaml(f.name)
        
        # Assertions
        assert isinstance(config.data, SFTDataConfig)
        assert config.data.input_file == 'test.jsonl'
        assert config.data.prompt_format == '{prompt}'
        assert config.data.response_format == '{response}'
        assert config.model.architecture == 'llama'
        assert config.training.batch_size == 4

# Test SFTDataManager
def test_sft_data_manager_batch_generation():
    # Mock dependencies
    tokenizer = MagicMock()
    tokenizer.tokenize.return_value = [1, 2, 3, 4, 5]
    tokenizer.VOCAB_SIZE = 100
    
    config = MagicMock()
    config.input_file = 'test.jsonl'
    config.validation_file = None
    config.prompt_format = '{prompt}'
    config.response_format = '{response}'
    
    # Mock data loading
    with patch('sft.SFTDataManager._load_file'):
        with patch('sft.SFTDataManager._create_batch', return_value={'input_ids': mx.array([1, 2, 3]), 'targets': mx.array([2, 3, 4])}):
            data_manager = SFTDataManager(config, tokenizer, batch_size=2)
            data_manager.data = [{'prompt': 'test prompt', 'response': 'test response'}] * 10
            
            # Test batch generation
            batch = data_manager.generate_batch(0)
            assert 'input_ids' in batch
            assert 'targets' in batch

# Test SFTTrainer
@pytest.mark.parametrize("pretrained_model_path", [None, "pretrained_model"])
def test_sft_trainer_initialization(pretrained_model_path):
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+') as f:
        yaml.dump({
            'data': {
                'input_file': 'test.jsonl',
                'prompt_format': '{prompt}',
                'response_format': '{response}',
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
        with patch('sft.SFTTrainer.setup_model'):
            with patch('sft.SFTTrainer.setup_training'):
                with patch('sft.SFTTrainer.setup_logging'):
                    with patch('train.TokenizerManager'):
                        with patch('os.makedirs'):
                            # Test initialization
                            trainer = SFTTrainer(f.name, pretrained_model_path=pretrained_model_path)
                            
                            # Assertions
                            assert trainer.config is not None
                            assert trainer.pretrained_model_path == pretrained_model_path