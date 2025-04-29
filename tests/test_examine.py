import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from examine import load_jsonl, count_tokens, main

# Test load_jsonl function
def test_load_jsonl():
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
        loaded_data = load_jsonl(f.name)
        
        # Verify
        assert loaded_data == test_data

# Test count_tokens function
def test_count_tokens():
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
        
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 2, 3, 4, 5]  # 5 tokens per text
        mock_tokenizer.encode.return_value = mock_encoding
        
        # Mock Tokenizer.from_file
        with patch('examine.Tokenizer.from_file', return_value=mock_tokenizer):
            # Test function
            with tempfile.NamedTemporaryFile(suffix='.json', mode='w+') as tokenizer_file:
                token_count = count_tokens(f.name, tokenizer_file.name)
                
                # Verify
                assert token_count == 10  # 2 texts * 5 tokens

# Test main function
def test_main():
    # Mock command line arguments
    with patch('argparse.ArgumentParser.parse_args', 
               return_value=MagicMock(
                   count_tokens=True,
                   data_path='test.jsonl',
                   tokenizer_path='tokenizer.json'
               )):
        # Mock count_tokens function
        with patch('examine.count_tokens', return_value=100):
            # Test function
            main()