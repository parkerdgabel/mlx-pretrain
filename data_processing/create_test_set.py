import json
import random
from pathlib import Path

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Path to the validation file
    val_path = Path('val.jsonl')
    test_set_path = Path('validation_test_set.jsonl')
    
    if not val_path.exists():
        raise ValueError(f"Validation file not found: {val_path}")
    
    # Load all sequences from val.jsonl
    with open(val_path, 'r') as f:
        val_data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(val_data)} sequences from validation set")
    
    # Sample 2048 random sequences
    test_set = random.sample(val_data, 2048)
    print(f"Sampled {len(test_set)} sequences for the test set")
    
    # Write the test set to a new file
    with open(test_set_path, 'w') as f:
        for item in test_set:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created fixed test set at {test_set_path}")
    
    # Verify the test set
    with open(test_set_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"Verification: test set contains {len(test_data)} sequences")

if __name__ == "__main__":
    main()