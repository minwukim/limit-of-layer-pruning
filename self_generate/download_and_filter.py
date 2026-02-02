import argparse
import csv
import json
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

# Increase CSV field size limit just in case
csv.field_size_limit(sys.maxsize)

def main():
    parser = argparse.ArgumentParser(description="Process dataset to CSV with length filtering")
    parser.add_argument("--dataset_name", type=str, default="open-thoughts/OpenThoughts3-1.2M", help="Dataset to load")
    parser.add_argument("--output_file", type=str, default="/scratch/ss13750/nnsight/openthoughts/data.csv", help="Output CSV path")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of samples to save")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_length", type=int, default=8192, help="Max token length for filtering")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer model name")
    parser.add_argument("--buffer_size", type=int, default=500000, help="Buffer size for shuffle")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading dataset: {args.dataset_name} (streaming)")
    
    # Load dataset in streaming mode
    try:
        ds = load_dataset(args.dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset {args.dataset_name}: {e}")
        return

    # Shuffle with seed (streaming shuffle uses a buffer). 
    print(f"Shuffling with seed {args.seed} and buffer size {args.buffer_size}...")
    ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    print(f"Processing until {args.num_samples} valid samples collected...")
    
    count = 0
    scanned = 0
    writer = None
    csvfile = None
    
    try:
        for sample in ds:
            if count >= args.num_samples:
                break
            
            scanned += 1
            if scanned % 10000 == 0:
                print(f"Scanned {scanned} samples. Saved {count} so far.")

            question = None
            response = None
            
            # Extract question/response based on dataset structure
            if 'conversations' in sample:
                # OpenThoughts format
                for turn in sample['conversations']:
                    if turn.get('from') == 'human':
                        question = turn.get('value')
                    else:
                        response = turn.get('value')
            elif 'messages' in sample:
                # Dolci / Standard format
                for msg in sample['messages']:
                    if msg.get('role') == 'user':
                        question = msg.get('content')
                    elif msg.get('role') == 'assistant':
                        response = msg.get('content')
            elif 'instruction' in sample and 'output' in sample:
                # Alpaca format
                instr = sample.get('instruction', '')
                inp = sample.get('input', '')
                if inp:
                    question = f"{instr}\n{inp}"
                else:
                    question = instr
                response = sample.get('output', '')
            
            if not question or not response:
                continue

            # Remove NUL bytes which crash CSV writer
            question = question.replace('\0', '')
            response = response.replace('\0', '')

            # Filter by length
            try:
                # Calculate total length using chat template
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                
                if len(token_ids) > args.max_length:
                    continue
            except Exception as e:
                print(f"Tokenization error on sample {scanned}: {e}")
                continue

            # Prepare row
            row = sample.copy()
            row['question'] = question
            row['response'] = response
            
            # Initialize writer lazily to know headers
            if writer is None:
                fieldnames = list(row.keys())
                # Ensure question/response are in fieldnames
                if 'question' not in fieldnames: fieldnames.append('question')
                if 'response' not in fieldnames: fieldnames.append('response')
                
                csvfile = open(args.output_file, 'w', newline='', encoding='utf-8')
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, escapechar='\\')
                writer.writeheader()

            try:
                writer.writerow(row)
                count += 1
                
                if count % 1000 == 0:
                    print(f"Saved {count} samples...")
            except csv.Error as e:
                print(f"Skipping row due to CSV error: {e}")
            except Exception as e:
                print(f"Skipping row due to unexpected error: {e}")

    finally:
        if csvfile:
            csvfile.close()

    print(f"Finished. Scanned {scanned} samples. Saved {count} samples to {args.output_file}")

if __name__ == "__main__":
    print("Starting...")
    main()
