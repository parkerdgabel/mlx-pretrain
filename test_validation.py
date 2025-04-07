import argparse
import json
import mlx.core as mx
from train import Trainer
from pathlib import Path
from tqdm import tqdm
from generate_lite import generate_lite, beam_search
import random
from mlx_lm.sample_utils import make_sampler, make_logits_processors

def main():
    parser = argparse.ArgumentParser(description='Test model on fixed validation test set')
    parser.add_argument('--run', type=str, default='OEIS-4M',
                       help='Name of the training run to use')
    parser.add_argument('--test-file', type=str, default='validation_test_set.jsonl',
                        help='Path to fixed validation test set')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (0.0 for deterministic)')
    parser.add_argument('--beam-search', action='store_true', 
                        help='Use beam search for generation instead of sampling')
    parser.add_argument('--num-beams', type=int, default=4,
                        help='Number of beams to use for beam search (only used if --beam-search is set)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to save detailed test results (JSON)')
    args = parser.parse_args()

    # Load run configuration and initialize trainer
    config_path = Path('runs') / args.run / 'config.yaml'
    if not config_path.exists():
        raise ValueError(f"Config not found for run: {args.run}")
    
    trainer = Trainer(str(config_path), for_training=False)
    
    # Load the final checkpoint
    checkpoint_path = Path('runs') / args.run / 'checkpoints' / 'step_final.safetensors'
    if not checkpoint_path.exists():
        raise ValueError(f"Final checkpoint not found for run: {args.run}")
    
    trainer.model.load_weights(str(checkpoint_path))
    
    # Setup generation parameters
    sampler = make_sampler(temp=args.temperature)
    logits_processors = make_logits_processors()
    
    # Load test data
    test_path = Path(args.test_file)
    if not test_path.exists():
        raise ValueError(f"Test file not found: {args.test_file}")
    
    with open(test_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    total_correct = 0
    total_tested = 0
    
    print(f"Testing sequence prediction on model: {args.run}")
    print(f"Using fixed test set: {args.test_file} ({len(test_data)} sequences)")
    print(f"Generation method: {'Beam search' if args.beam_search else 'Sampling'} (Temperature: {args.temperature})")
    print("-" * 60)
    
    results = []
    
    # Test each sequence in the fixed test set
    for i, sample in enumerate(tqdm(test_data, desc="Testing")):
        sequence_str = sample['text']
        
        # Parse the sequence
        sequence = [int(x) for x in sequence_str.split(',')]
        
        if len(sequence) < 2:
            print(f"Skipping sequence {i} - too short: {sequence_str}")
            continue
        
        # Use all but last term as input, and last term as expected output
        input_seq = sequence[:-1]
        expected_next = sequence[-1]
        
        # Format the input as a string like OEIS format
        input_str = ','.join(str(n) for n in input_seq) + ','
        
        # Tokenize the input
        tokens = [trainer.tokenizer.BOS_TOKEN] + trainer.tokenizer.tokenize(input_str)
        
        # Generate the next token, using comma as stop token
        comma_token = trainer.tokenizer.tokenize(",")[0]
        generated = None
        try:
            if args.beam_search:
                generated = beam_search(
                    trainer.model,
                    mx.array(tokens),
                    max_tokens=100,
                    n_beams=args.num_beams,
                    stop_tokens=[comma_token, trainer.tokenizer.EOS_TOKEN, trainer.tokenizer.BOS_TOKEN, trainer.tokenizer.PAD_TOKEN],
                )[0][0]
            else:
                generated, _ = generate_lite(
                    trainer.model,
                    mx.array(tokens),
                    max_tokens=100,  
                    sampler=sampler,
                    stop_tokens=[comma_token, trainer.tokenizer.EOS_TOKEN, trainer.tokenizer.BOS_TOKEN, trainer.tokenizer.PAD_TOKEN],
                )
                generated = generated.tolist()[:-1]
            # Detokenize to get the prediction
            prediction_str = trainer.tokenizer.detokenize(generated)
            
            # Clean up and convert to integer
            try:
                predicted_next = int(prediction_str)
                is_correct = (predicted_next == expected_next)
                
                # Track results
                total_tested += 1
                if is_correct:
                    total_correct += 1
                
                # Store detailed result for this test case
                results.append({
                    'sequence': input_seq,
                    'expected': expected_next,
                    'predicted': predicted_next,
                    'correct': is_correct
                })
            except ValueError:
                print(f"Could not parse prediction: '{prediction_str}' for sequence: {input_seq}")
        except Exception as e:
            print(f"Error processing sequence {i}: {e}")
            continue
            
    # Calculate accuracy
    accuracy = (total_correct / total_tested) * 100 if total_tested > 0 else 0
    
    # Print results
    print("\nValidation Results:")
    print(f"Total test cases: {total_tested}")
    print(f"Correct predictions: {total_correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    """
    # Print some example results (correct and incorrect)
    print("\nSample Correct Predictions:")
    correct_samples = [r for r in results if r['correct']]
    random.shuffle(correct_samples)  # Shuffle to get a random sample of correct predictions
    correct_examples = correct_samples[:10]
    for i, example in enumerate(correct_examples, 1):
        seq_str = ','.join(str(n) for n in example['sequence'])
        print(f"{i}. {seq_str} → {example['predicted']} ✓")
    
    print("\nSample Incorrect Predictions:")
    incorrect_samples = [r for r in results if not r['correct']]
    random.shuffle(incorrect_samples)  # Shuffle to get a random sample of incorrect predictions
    incorrect_examples = incorrect_samples[:10]
    for i, example in enumerate(incorrect_examples, 1):
        seq_str = ','.join(str(n) for n in example['sequence'])
        print(f"{i}. {seq_str} → {example['predicted']} (expected {example['expected']}) ✗")
    """
    # Save detailed results if an output file is specified
    if args.output_file:
        output_path = Path(args.output_file)
        test_results = {
            'model': args.run,
            'test_file': args.test_file,
            'temperature': args.temperature,
            'beam_search': args.beam_search,
            'num_beams': args.num_beams if args.beam_search else None,
            'total_tested': total_tested,
            'total_correct': total_correct,
            'accuracy': accuracy,
            'results': results
        }
        with open(output_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()