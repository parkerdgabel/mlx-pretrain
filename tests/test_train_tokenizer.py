import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from train_tokenizer import load_config, load_jsonl_texts, batch_iterator, train_tokenizer, main

# Test load_config function
def test_load_config():
    # Create test config
    test_config = {
        'data': {
            'input_file': 'test.jsonl',
            'tokenizer': {
                'special_tokens': {
                    'bos': '<s>',
                    'eos': '</s>'
                }
            }
        },
        'tokenizer': {
            'vocab_size': 10000,
            'output_dir': 'tokenizer'
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w+') as f:
        yaml.dump(test_config, f)
        f.flush()
        
        # Test function
        loaded_config = load_config(f.name)
        
        # Verify
        assert loaded_config == test_config

# Test load_jsonl_texts function
def test_load_jsonl_texts():
    # Create test data
    test_data = [
        {"text": "This is a test."},
        {"text": "Another test example."}
    ]
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jsonl', mode='w+') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
        f.flush()
        
        # Test function
        loaded_texts = load_jsonl_texts(f.name)
        
        # Verify
        assert loaded_texts == ["This is a test.", "Another test example."]

# Test batch_iterator function
def test_batch_iterator():
    # Create test data
    test_texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
    
    # Test function with batch size 2
    batches = list(batch_iterator(test_texts, batch_size=2))
    
    # Verify
    assert len(batches) == 3
    assert batches[0] == ["Text 1", "Text 2"]
    assert batches[1] == ["Text 3", "Text 4"]
    assert batches[2] == ["Text 5"]

# Test train_tokenizer function
def test_train_tokenizer():
    # Create test config
    test_config = {
        'data': {
            'input_file': 'test.jsonl',
            'tokenizer': {
                'special_tokens': {
                    'bos': '<s>',
                    'eos': '</s>'
                }
            }
        },
        'tokenizer': {
            'vocab_size': 10000,
            'output_dir': 'tokenizer'
        }
    }
    
    # Mock dependencies
    with patch('train_tokenizer.Tokenizer'):
        with patch('train_tokenizer.BPE'):
            with patch('train_tokenizer.BpeTrainer'):
                with patch('train_tokenizer.load_jsonl_texts', return_value=["Test text 1", "Test text 2"]):
                    with patch('os.makedirs'):
                        # Test function
                        tokenizer = train_tokenizer(test_config)
                        
                        # Verify
                        assert tokenizer is not None

# Test main function
def test_main():
    # Mock command line arguments
    with patch('argparse.ArgumentParser.parse_args', 
               return_value=MagicMock(
                   config='config.yaml'
               )):
        # Mock dependencies
        with patch('train_tokenizer.load_config', return_value={}):
            with patch('train_tokenizer.train_tokenizer'):
                # Test function
                main()