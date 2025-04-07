import argparse
import mlx.core as mx
from train import Trainer
from pathlib import Path
from tqdm import tqdm
from generate_lite import generate_lite,beam_search
import random
from mlx_lm.sample_utils import make_sampler, make_logits_processors
def generate_arithmetic_sequence(a, b, length=9):
    """Generate an arithmetic sequence of the form a*n + b"""
    return [a * n + b for n in range(1, length + 1)]

def main():
    parser = argparse.ArgumentParser(description='Test model on arithmetic sequences')
    parser.add_argument('--run', type=str, default='OEIS-4M',
                       help='Name of the training run to use')
    parser.add_argument('--max-a', type=int, default=1000,
                       help='Maximum value for parameter a')
    parser.add_argument('--max-b', type=int, default=1000,
                       help='Maximum value for parameter b')
    parser.add_argument('--min-a', type=int, default=1,
                       help='Minimum value for parameter a')
    parser.add_argument('--min-b', type=int, default=0,
                       help='Minimum value for parameter b')
    parser.add_argument('--terms', type=int, default=9,
                       help='Number of terms to provide to the model')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (0.0 for deterministic)')
    parser.add_argument('--num-tests', type=int, default=1000,
                       help='Number of random tests to run')
    parser.add_argument('--beam-search', action='store_true', 
                        help='Use beam search for generation instead of sampling')
    parser.add_argument('--num-beams', type=int, default=4,
                        help='Number of beams to use for beam search (only used if --beam-search is set)')
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
    sampler = make_sampler(temp=args.temperature)  # Use a low temperature for deterministic generation
    logits_processors = make_logits_processors()
    
    total_correct = 0
    total_tested = 0
    
    print(f"Testing arithmetic sequences on model: {args.run}")
    print(f"Parameters range: a={args.min_a}-{args.max_a}, b={args.min_b}-{args.max_b}")
    print(f"Providing {args.terms} terms to predict the next term")
    print(f"Running {args.num_tests} random tests")
    print("-" * 60)
    
    results = []
    
    # Generate random test cases
    for _ in tqdm(range(args.num_tests), desc="Testing"):
        # Generate random a and b values within the specified ranges
        a = random.randint(args.min_a, args.max_a)
        b = random.randint(args.min_b, args.max_b)
        
        # Generate the sequence
        sequence = generate_arithmetic_sequence(a, b, length=args.terms + 1)
        
        # Prepare input: first 'terms' elements of the sequence
        input_seq = sequence[:args.terms]
        expected_next = sequence[args.terms]
        
        # Format the input as a string like OEIS format
        input_str = ','.join(str(n) for n in input_seq) + ','
        
        # Tokenize the input (remove the last comma and space)
        tokens = [trainer.tokenizer.BOS_TOKEN] + trainer.tokenizer.tokenize(input_str)
        
        # Generate the next token with generate_lite, using comma as stop token
        comma_token = trainer.tokenizer.tokenize(",")[0]
        generated = None
        if args.beam_search:
            generated = beam_search(
                trainer.model,
                mx.array(tokens),
                max_tokens=10,  # Limit to 10 tokens maximum
                n_beams=args.num_beams,
                stop_tokens=[comma_token, trainer.tokenizer.EOS_TOKEN],
            )[0][0]
        else:
            generated, _ = generate_lite(
                trainer.model,
                mx.array(tokens),
                max_tokens=10,  # Limit to 10 tokens maximum
                sampler=sampler,  # Use the sampler defined above
                stop_tokens=[comma_token, trainer.tokenizer.EOS_TOKEN],
            )
            generated = generated.tolist()[:-1]
        # Detokenize to get the prediction
        prediction_str = trainer.tokenizer.detokenize(generated)
        # Clean up the prediction string - try to extract just the number
        # Convert to integer
        predicted_next = int(prediction_str)
        is_correct = (predicted_next == expected_next)
        
        # Track results
        total_tested += 1
        if is_correct:
            total_correct += 1
        
        # Store detailed result for this test case
        results.append({
            'a': a, 
            'b': b, 
            'sequence': input_seq,
            'expected': expected_next,
            'predicted': predicted_next,
            'correct': is_correct
        })
            
    # Calculate accuracy
    accuracy = (total_correct / total_tested) * 100 if total_tested > 0 else 0
    
    # Print results
    print("\nTest Results:")
    print(f"Total test cases: {total_tested}")
    print(f"Correct predictions: {total_correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Print some example results (first 5 correct and first 5 incorrect)
    print("\nSample Correct Predictions:")
    s = [r for r in results if r['correct']]
    random.shuffle(s)  # Shuffle to get a random sample of correct predictions
    correct_examples = s[:10]
    for i, example in enumerate(correct_examples, 1):
        print(f"{i}. a={example['a']}, b={example['b']}: {example['sequence']} → {example['predicted']} ✓")
    
    print("\nSample Incorrect Predictions:")
    #incorrect_examples = [r for r in results if not r['correct']][:10]
    s = [r for r in results if not r['correct']]
    random.shuffle(s)  # Shuffle to get a random sample of incorrect predictions
    incorrect_examples = s[:10]
    for i, example in enumerate(incorrect_examples, 1):
        print(f"{i}. a={example['a']}, b={example['b']}: {example['sequence']} → {example['predicted']} (expected {example['expected']}) ✗")

if __name__ == "__main__":
    main()