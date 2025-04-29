import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from dataset import process_dataset, save_to_jsonl, parse_args

# Test save_to_jsonl function
def test_save_to_jsonl():
    test_data = [
        {"text": "This is a test."},
        {"text": "Another test example."}
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.jsonl', mode='w+') as f:
        save_to_jsonl(test_data, f.name)
        
        # Read back and verify
        f.seek(0)
        lines = f.readlines()
        assert len(lines) == 2
        
        loaded_data = [json.loads(line) for line in lines]
        assert loaded_data == test_data

# Test process_dataset function
def test_process_dataset():
    # Mock dataset loading
    mock_dataset = {
        'train': MagicMock(),
        'validation': MagicMock()
    }
    
    mock_dataset['train'].features = {'text': None}
    mock_dataset['validation'].features = {'text': None}
    
    # Create sample data
    train_data = [{'text': f'Train text {i}'} for i in range(10)]
    val_data = [{'text': f'Val text {i}'} for i in range(5)]
    
    mock_dataset['train'].__iter__ = lambda self: iter(train_data)
    mock_dataset['train'].__len__ = lambda self: len(train_data)
    mock_dataset['train'].select = lambda indices: [train_data[i] for i in indices]
    
    mock_dataset['validation'].__iter__ = lambda self: iter(val_data)
    mock_dataset['validation'].__len__ = lambda self: len(val_data)
    mock_dataset['validation'].select = lambda indices: [val_data[i] for i in indices]
    
    with patch('dataset.load_dataset', return_value=mock_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_output = os.path.join(tmpdir, 'train.jsonl')
            val_output = os.path.join(tmpdir, 'val.jsonl')
            
            # Test function
            process_dataset(
                dataset_name='test_dataset',
                train_split='train',
                val_split='validation',
                text_field='text',
                train_output=train_output,
                val_output=val_output
            )
            
            # Verify outputs
            assert os.path.exists(train_output)
            assert os.path.exists(val_output)
            
            # Check content
            with open(train_output, 'r') as f:
                train_lines = f.readlines()
                assert len(train_lines) == 10
            
            with open(val_output, 'r') as f:
                val_lines = f.readlines()
                assert len(val_lines) == 5

# Test argument parsing
def test_parse_args():
    with patch('argparse.ArgumentParser.parse_args', 
               return_value=MagicMock(
                   dataset='test_dataset',
                   name=None,
                   train_split='train',
                   val_split='validation',
                   text_field='text',
                   train_output='train.jsonl',
                   val_output='val.jsonl',
                   limit=None,
                   val_ratio=0.1
               )):
        args = parse_args()
        assert args.dataset == 'test_dataset'
        assert args.train_split == 'train'
        assert args.val_split == 'validation'
        assert args.text_field == 'text'
        assert args.train_output == 'train.jsonl'
        assert args.val_output == 'val.jsonl'
        assert args.limit is None
        assert args.val_ratio == 0.1