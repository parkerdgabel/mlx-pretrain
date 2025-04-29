import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import mlx.core as mx
import numpy as np

# Import the functions to test
from generate import main as generate_main
from generate_lite import generate_lite, beam_search, generate_step

# Test generate_lite function
def test_generate_lite():
    # Mock model
    model = MagicMock()
    model.return_value = {'logits': mx.array(np.random.randn(1, 100))}
    
    # Mock input
    prompt = mx.array([1, 2, 3])
    
    # Mock sampler
    def mock_sampler(logits):
        return mx.array([4])  # Always return token 4
    
    # Test function
    with patch('generate_lite.maybe_quantize_kv_cache', return_value=None):
        output, score = generate_lite(
            model=model,
            prompt=prompt,
            max_tokens=5,
            sampler=mock_sampler,
            verbose=False
        )
        
        # Verify output
        assert len(output) > len(prompt)
        assert isinstance(output, mx.array)
        assert isinstance(score, float)

# Test beam_search function
def test_beam_search():
    # Mock model
    model = MagicMock()
    model.return_value = {'logits': mx.array(np.random.randn(1, 100))}
    
    # Mock input
    input_tokens = mx.array([1, 2, 3])
    
    # Test function
    with patch('generate_lite.maybe_quantize_kv_cache', return_value=None):
        output = beam_search(
            model=model,
            input_tokens=input_tokens,
            max_tokens=5,
            n_beams=2,
            verbose=False
        )
        
        # Verify output
        assert len(output) > 0
        assert isinstance(output, list)

# Test generate_step function
def test_generate_step():
    # Mock model
    model = MagicMock()
    model.return_value = {'logits': mx.array(np.random.randn(1, 100))}
    
    # Mock input
    prompt = mx.array([1, 2, 3])
    
    # Test function
    with patch('generate_lite.maybe_quantize_kv_cache', return_value=None):
        tokens, scores = generate_step(
            prompt=prompt,
            model=model,
            max_tokens=5
        )
        
        # Verify output
        assert len(tokens) > 0
        assert len(scores) > 0
        assert isinstance(tokens, list)
        assert isinstance(scores, list)

# Test main function in generate.py
def test_generate_main():
    # Create a temporary config file
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / 'test_run'
        os.makedirs(run_dir / 'checkpoints', exist_ok=True)
        
        # Create mock config
        with open(run_dir / 'config.yaml', 'w') as f:
            f.write("""
            data:
              tokenizer:
                special_tokens:
                  bos: '<s>'
                  eos: '</s>'
                  pad: '<pad>'
                  unk: '<unk>'
            model:
              architecture: 'llama'
            """)
        
        # Create mock checkpoint
        with open(run_dir / 'checkpoints' / 'step_final_model.safetensors', 'w') as f:
            f.write("mock checkpoint")
        
        # Mock dependencies
        with patch('generate.Trainer'):
            with patch('generate.generate_lite', return_value=(mx.array([1, 2, 3, 4]), 0.5)):
                with patch('sys.argv', ['generate.py', '--run', str(run_dir), '--prompt', 'Test prompt']):
                    with patch('os.path.exists', return_value=True):
                        # Test function
                        with pytest.raises(SystemExit) as e:
                            generate_main()