import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def run_test(terms, beam_search=False):
    """Run the arithmetic test with specified parameters and return accuracy."""
    cmd = ["python", "test_arithmetic.py", "--terms", str(terms)]
    
    if beam_search:
        cmd.extend(["--beam-search"])
    
    # Run the test and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse the output to extract accuracy
    for line in result.stdout.split('\n'):
        if line.startswith("Accuracy:"):
            accuracy = float(line.split(':')[1].strip().rstrip('%'))
            return accuracy / 100  # Convert percentage to proportion
    
    raise ValueError(f"Failed to extract accuracy from output: {result.stdout}")

def calculate_error_bar(p, n):
    """Calculate 95% confidence interval using normal approximation."""
    return 1.96 * np.sqrt((p * (1 - p)) / n)

def main():
    num_tests = 1000  # Number of tests per configuration
    results = {}
    
    print("Running arithmetic sequence tests:")
    
    # Test for terms 2-9 with both greedy and beam search
    for terms in range(2, 21):
        print(f"\nTesting with {terms} terms:")
        
        # Greedy search (temperature = 0)
        print("  Running greedy search...")
        greedy_acc = run_test(terms, beam_search=False)
        greedy_err = calculate_error_bar(greedy_acc, num_tests)
        
        # Beam search
        print("  Running beam search...")
        beam_acc = run_test(terms, beam_search=True)
        beam_err = calculate_error_bar(beam_acc, num_tests)
        
        results[terms] = {
            'greedy': {
                'accuracy': greedy_acc,
                'error': greedy_err
            },
            'beam': {
                'accuracy': beam_acc,
                'error': beam_err
            }
        }
        
        print(f"  Greedy accuracy: {greedy_acc:.4f} ± {greedy_err:.4f}")
        print(f"  Beam accuracy: {beam_acc:.4f} ± {beam_err:.4f}")
    
    # Save results to file
    output_file = Path('arithmetic_test_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Plot results
    plot_results(results)

def plot_results(results):
    terms = list(results.keys())
    terms.sort()  # Ensure terms are in ascending order
    
    greedy_acc = [results[t]['greedy']['accuracy'] for t in terms]
    greedy_err = [results[t]['greedy']['error'] for t in terms]
    
    beam_acc = [results[t]['beam']['accuracy'] for t in terms]
    beam_err = [results[t]['beam']['error'] for t in terms]
    
    plt.figure(figsize=(10, 6))
    
    # Plot greedy search results
    plt.errorbar(terms, greedy_acc, yerr=greedy_err, fmt='o-', label='Greedy Search', capsize=5)
    
    # Plot beam search results
    plt.errorbar(terms, beam_acc, yerr=beam_err, fmt='s-', label='Beam Search', capsize=5)
    
    plt.xlabel('Number of Terms')
    plt.ylabel('Accuracy')
    plt.title('Arithmetic Sequence Prediction Accuracy')
    plt.xticks(terms)
    plt.ylim(0, 1.05)  # Set y-axis limits
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.savefig('graphs/arithmetic_test_results.png')
    print("Plot saved to arithmetic_test_results.png")
    
    # Display plot
    plt.show()

if __name__ == "__main__":
    main()