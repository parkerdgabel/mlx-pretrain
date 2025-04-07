import json

with open("train.jsonl", "r") as f:
    max_length = 0
    for line in f:
        d = json.loads(line)
        text_length = len(d["text"])
        if text_length > max_length:
            max_length = text_length
    print(f"Maximum length of any text in train.jsonl: {max_length}")