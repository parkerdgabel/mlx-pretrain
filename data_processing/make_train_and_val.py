# Load oeis_processed, shuffle order, and split into 90% train and 10% validation

import json
import random

# Load the processed OEIS data
with open('oeis_processed.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
    random.shuffle(data)  # Shuffle the data to ensure randomness
    train_split_index = int(len(data) * 0.9)  # 90% for training
    train_data = data[:train_split_index]
    val_data = data[train_split_index:]  # Remaining 10% for validation
    with open('train.jsonl', 'w') as train_f:
        for item in train_data:
            train_f.write(json.dumps(item) + "\n")
    with open('val.jsonl', 'w') as val_f:
        for item in val_data:
            val_f.write(json.dumps(item) + "\n")