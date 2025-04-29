import pytest
import tempfile
import yaml
import json
import os
import mlx.core as mx
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def temp_config_file():
    """Create a temporary YAML config file."""
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+') as f:
        yield f

@pytest.fixture
def temp_jsonl_file():
    """Create a temporary JSONL file."""
    with tempfile.NamedTemporaryFile(suffix='.jsonl', mode='w+') as f:
        yield f

@pytest.fixture
def basic_config_dict():
    """Return a basic configuration dictionary."""
    return {
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
    }

@pytest.fixture
def basic_config_file(temp_config_file, basic_config_dict):
    """Create a config file with basic configuration."""
    yaml.dump(basic_config_dict, temp_config_file)
    temp_config_file.flush()
    return temp_config_file.name

@pytest.fixture
def sample_jsonl_data():
    """Return sample data for JSONL files."""
    return [
        {"text": "This is a test document."},
        {"text": "Another test example with more text."},
        {"text": "A third document for testing purposes."}
    ]

@pytest.fixture
def sample_jsonl_file(temp_jsonl_file, sample_jsonl_data):
    """Create a JSONL file with sample data."""
    for item in sample_jsonl_data:
        temp_jsonl_file.write(json.dumps(item).encode('utf-8') + b'\n')
    temp_jsonl_file.flush()
    return temp_jsonl_file.name

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.BOS_TOKEN = '<s>'
    tokenizer.EOS_TOKEN = '</s>'
    tokenizer.PAD_TOKEN = '<pad>'
    tokenizer.UNK_TOKEN = '<unk>'
    tokenizer.VOCAB_SIZE = 100
    tokenizer.tokenize.return_value = [1, 2, 3, 4, 5]
    tokenizer.detokenize.return_value = "Test text"
    return tokenizer

@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.return_value = {'logits': mx.array(np.random.randn(2, 5, 100))}
    return model