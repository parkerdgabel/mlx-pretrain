import json

with open('oeis.jsonl', 'r') as f:
    with open("oeis_processed.jsonl", 'w') as out_f:
        for line in f:
            seq = json.loads(line)['sequence']
            # Join sequence w/ commas
            seq_str = ",".join(seq)
            # Start w/ colon and terminate with semicolon
            seq_str = seq_str
            # Write the processed sequence to the output file
            out_f.write(json.dumps({"text": seq_str}) + "\n")
            


    