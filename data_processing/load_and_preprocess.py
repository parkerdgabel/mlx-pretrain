import pandas as pd

# Load oeis.parquet and convert to jsonl

def load_and_convert_to_jsonl(parquet_file, jsonl_file):
    """
    Load a parquet file and convert it to a jsonl file.
    
    :param parquet_file: Path to the input parquet file.
    :param jsonl_file: Path to the output jsonl file.
    """
    # Load the parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)
    
    # Convert the DataFrame to JSON lines format and save to file
    df.to_json(jsonl_file, orient='records', lines=True)

if __name__ == "__main__":
    # Define the input and output file paths
    parquet_file = 'oeis.parquet'
    jsonl_file = 'oeis.jsonl'
    
    # Load and convert the parquet file to jsonl
    load_and_convert_to_jsonl(parquet_file, jsonl_file)