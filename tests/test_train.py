import pytest
import os
import tempfile
import yaml
import mlx.core as mx
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from train import (
    Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig, 
    SystemConfig, ResumeConfig, TokenizerManager, DataManager, 
    OptimizationManager, Trainer, CheckpointManager, VLMEvaluator,
    filter_valid_args
)

# Test Config loading
def test_config_from_yaml():
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+') as f:
        yaml.dump({
            'data': {'input_file': 'test.jsonl'},
            'model': {'architecture': 'llama'},
            'training': {'batch_size': 4},
            'logging': {'log_interval': 10},
            'system': {'seed': 42},
        }, f)
        f.flush()
        
        # Test loading
        config = Config.from_yaml(f.name)
        
        # Assertions
        assert isinstance(config.data, DataConfig)
        assert config.data.input_file == 'test.jsonl'
        assert config.model.architecture == 'llama'
        assert config.training.batch_size == 4
        assert config.logging.log_interval == 10
        assert config.system.seed == 42

# Test TokenizerManager
def test_tokenizer_manager_initialization():
    config = MagicMock()
    config.tokenizer = {
        'special_tokens': {
            'bos': '<s>',
            'eos': '</s>',
            'pad': '<pad>',
            'unk': '<unk>'
        }
    }
    
    with patch('os.path.exists', return_value=True):
        with patch('train.TokenizerManager.use_external_tokenizer'):
            tokenizer = TokenizerManager(config)
            assert tokenizer.BOS_TOKEN == '<s>'
            assert tokenizer.EOS_TOKEN == '</s>'
            assert tokenizer.PAD_TOKEN == '<pad>'
            assert tokenizer.UNK_TOKEN == '<unk>'

# Test DataManager
def test_data_manager_batch_generation():
    # Mock dependencies
    tokenizer = MagicMock()
    tokenizer.tokenize.return_value = [1, 2, 3, 4, 5]
    tokenizer.VOCAB_SIZE = 100
    
    config = MagicMock()
    config.input_file = 'test.jsonl'
    config.validation_file = None
    config.preprocessing = {'max_context_size': 10}
    
    # Mock data loading
    with patch('train.DataManager._load_file'):
        with patch('train.DataManager._create_batch', return_value={'input_ids': mx.array([1, 2, 3]), 'targets': mx.array([2, 3, 4])}):
            data_manager = DataManager(config, tokenizer, batch_size=2)
            data_manager.docs = [{'text': 'test text'}] * 10
            
            # Test batch generation
            batch = data_manager.generate_batch(0)
            assert 'input_ids' in batch
            assert 'targets' in batch

# Test OptimizationManager
def test_optimization_manager():
    config = MagicMock()
    config.optimizer = 'adamw'
    config.learning_rate = 1e-4
    config.weight_decay = 0.01
    config.lr_schedule = 'cosine'
    config.warmup_steps = 100
    
    opt_manager = OptimizationManager(config, num_training_steps=1000)
    
    # Test scheduler creation
    schedule = opt_manager.create_scheduler()
    assert callable(schedule)
    
    # Test optimizer creation
    optimizer = opt_manager.create_optimizer(schedule)
    assert hasattr(optimizer, 'update')

# Test Trainer
@pytest.mark.parametrize("for_training", [True, False])
def test_trainer_initialization(for_training):
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+') as f:
        yaml.dump({
            'data': {
                'input_file': 'test.jsonl',
                'tokenizer': {
                    'path': 'tokenizer',
                    'special_tokens': {
                        'bos': '<s>',
                        'eos': '</s>',
                        'pad': '<pad>',
                        'unk': '<unk>'
                    }
                },
                'preprocessing': {'max_context_size': 10}
            },
            'model': {
                'architecture': 'llama',
                'dimensions': {
                    'hidden_size': 128,
                    'intermediate_size': 256,
                    'num_layers': 2
                },
                'attention': {'num_heads': 4},
                'normalization': {'rms_norm_eps': 1e-6},
                'misc': {
                    'attention_bias': False,
                    'mlp_bias': False,
                    'tie_word_embeddings': True
                },
                'rope': {
                    'scaling': None,
                    'theta': 10000
                }
            },
            'training': {
                'batch_size': 4,
                'optimizer': 'adamw',
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'lr_schedule': 'cosine',
                'warmup_steps': 100,
                'total_steps': 1000
            },
            'logging': {'log_interval': 10},
            'system': {'seed': 42},
        }, f)
        f.flush()
        
        # Mock dependencies
        with patch('train.Trainer.setup_model'):
            with patch('train.Trainer.setup_training'):
                with patch('train.Trainer.setup_logging'):
                    with patch('train.TokenizerManager'):
                        with patch('os.makedirs'):
                            # Test initialization
                            trainer = Trainer(f.name, for_training=for_training)
                            
                            # Assertions
                            assert trainer.config is not None
                            assert trainer.for_training == for_training

# Test compute_loss method
def test_trainer_compute_loss():
    trainer = MagicMock()
    trainer.compute_loss = Trainer.compute_loss
    
    # Mock model and inputs
    model = MagicMock()
    model.return_value = {'logits': mx.array(np.random.randn(2, 5, 100))}
    
    inputs = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    targets = mx.array([[2, 3, 4, 0], [6, 7, 8, 0]])
    
    # Test loss computation
    loss = trainer.compute_loss(trainer, model, inputs, targets)
    assert isinstance(loss, mx.array)