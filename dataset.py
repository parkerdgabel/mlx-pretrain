#!/usr/bin/env python
"""
Script to fetch datasets from Hugging Face and convert them to jsonl format for mlx-pretrain.

Usage:
    python dataset.py --dataset <dataset_name> [--train_split <train_split>] [--val_split <val_split>] 
                     [--text_field <text_field>] [--train_output <train_output>] [--val_output <val_output>]
                     [--limit <limit>] [--val_ratio <val_ratio>]

Example:
    python dataset.py --dataset "roneneldan/TinyStories" --text_field "text"
    python dataset.py --dataset "wikitext" --name "wikitext-103-v1" --train_split "train" --val_split "validation"
"""

import argparse
import json
import os
from typing import Optional, List, Dict, Any

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "The 'datasets' library is required but not installed. "
        "Please install it using: pip install datasets"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch dataset from Hugging Face and convert to jsonl")
    parser.add_argument("--dataset", required=True, help="Dataset name on Hugging Face")
    parser.add_argument("--name", help="Specific dataset configuration name")
    parser.add_argument("--train_split", default="train", help="Training split name (default: train)")
    parser.add_argument("--val_split", help="Validation split name (default: validation or test)")
    parser.add_argument("--text_field", default="text", help="Field containing text data (default: text)")
    parser.add_argument("--train_output", default="train.jsonl", help="Output path for training data (default: train.jsonl)")
    parser.add_argument("--val_output", default="val.jsonl", help="Output path for validation data (default: val.jsonl)")
    parser.add_argument("--limit", type=int, help="Limit number of examples (for testing)")
    parser.add_argument("--val_ratio", type=float, default=0.1, 
                        help="Ratio of training data to use for validation if no validation split exists (default: 0.1)")
    return parser.parse_args()


def save_to_jsonl(data: List[Dict[str, Any]], output_path: str) -> None:
    """Save data to a jsonl file."""
    print(f"Saving {len(data)} examples to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def process_dataset(
    dataset_name: str,
    config_name: Optional[str] = None,
    train_split: str = "train",
    val_split: Optional[str] = None,
    text_field: str = "text",
    train_output: str = "train.jsonl",
    val_output: str = "val.jsonl",
    limit: Optional[int] = None,
    val_ratio: float = 0.1
) -> None:
    """
    Load a dataset from Hugging Face and save it as jsonl files.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        config_name: Specific configuration of the dataset
        train_split: Name of the training split
        val_split: Name of the validation split
        text_field: Field containing the text data
        train_output: Output path for training data
        val_output: Output path for validation data
        limit: Limit number of examples (for testing)
        val_ratio: Ratio of training data to use for validation if no validation split exists
    """
    print(f"Loading dataset: {dataset_name}" + (f" ({config_name})" if config_name else ""))
    
    # Load the dataset
    dataset = load_dataset(dataset_name, config_name)
    
    # Check available splits
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")
    
    # Verify train split exists
    if train_split not in available_splits:
        raise ValueError(f"Training split '{train_split}' not found. Available splits: {available_splits}")
    
    # Get validation split
    if val_split is None:
        # Try common validation split names
        for split in ["validation", "valid", "dev", "test"]:
            if split in available_splits:
                val_split = split
                break
    
    # Process training data
    train_data = dataset[train_split]
    if limit:
        train_data = train_data.select(range(min(limit, len(train_data))))
    
    # Check if text field exists
    if text_field not in train_data.features:
        raise ValueError(f"Text field '{text_field}' not found. Available fields: {list(train_data.features.keys())}")
    
    # Convert to list of dictionaries with "text" field
    train_jsonl = []
    for item in train_data:
        if text_field in item and item[text_field]:
            train_jsonl.append({"text": item[text_field]})
    
    # Process validation data
    val_jsonl = []
    if val_split and val_split in available_splits:
        val_data = dataset[val_split]
        if limit:
            val_data = val_data.select(range(min(limit, len(val_data))))
        
        for item in val_data:
            if text_field in item and item[text_field]:
                val_jsonl.append({"text": item[text_field]})
    else:
        # Create validation set from training data if no validation split exists
        print(f"No validation split found. Creating validation set from {val_ratio*100}% of training data.")
        import random
        random.shuffle(train_jsonl)
        split_idx = int(len(train_jsonl) * (1 - val_ratio))
        val_jsonl = train_jsonl[split_idx:]
        train_jsonl = train_jsonl[:split_idx]
    
    # Save to jsonl files
    save_to_jsonl(train_jsonl, train_output)
    save_to_jsonl(val_jsonl, val_output)
    
    print(f"Dataset processing complete.")
    print(f"Training data: {len(train_jsonl)} examples saved to {train_output}")
    print(f"Validation data: {len(val_jsonl)} examples saved to {val_output}")


def main():
    args = parse_args()
    process_dataset(
        dataset_name=args.dataset,
        config_name=args.name,
        train_split=args.train_split,
        val_split=args.val_split,
        text_field=args.text_field,
        train_output=args.train_output,
        val_output=args.val_output,
        limit=args.limit,
        val_ratio=args.val_ratio
    )


if __name__ == "__main__":
    main()
