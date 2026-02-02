import argparse
import csv
import sys
import os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["VLLM_USE_V1"] = "0"

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

def get_stop_tokens(model_path, tokenizer):
    stop_tokens = set([tokenizer.eos_token_id])
    
    # Model specific stop tokens
    model_path_lower = model_path.lower()
    
    if "qwen" in model_path_lower:
        token = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if token is not None: stop_tokens.add(token)
        token = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if token is not None: stop_tokens.add(token)
        
    if "llama-3" in model_path_lower:
        # Llama 3/3.1 stop tokens
        for t in ["<|eot_id|>", "<|end_of_text|>"]:
            token = tokenizer.convert_tokens_to_ids(t)
            if token is not None: stop_tokens.add(token)
            
    if "mistral" in model_path_lower:
        # Mistral often uses </s> or [INST] depending on version, 
        # v0.3 uses standard tokens but sometimes specialized ones.
        for t in ["</s>", "[INST]", "<|file_separator|>"]:
            token = tokenizer.convert_tokens_to_ids(t)
            if token is not None: stop_tokens.add(token)
    
    if "deepseek" in model_path_lower:
        for t in ["<｜end▁of▁sentence｜>", "Assistant:"]:
            token = tokenizer.convert_tokens_to_ids(t)
            if token is not None: stop_tokens.add(token)

    if "gemma" in model_path_lower:
        for t in ["<end_of_turn>", "<eos>"]:
            token = tokenizer.convert_tokens_to_ids(t)
            if token is not None: stop_tokens.add(token)
    
    if "llama-2" in model_path_lower:
        for t in ["Question: ", "Answer: ", "Assistant: ", "You are a helpful assistant", "Comment: ", "Query: "]:
            token = tokenizer.convert_tokens_to_ids(t)
            if token is not None: stop_tokens.add(token)

    # Filter out None values
    stop_tokens = [t for t in stop_tokens if t is not None]
    return list(stop_tokens)

def main():
    parser = argparse.ArgumentParser(description="Generate responses using various models via vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or HF model ID")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of samples per batch")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", help="System prompt to use")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model context length")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    # Initialize LLM with max_model_len
    llm = LLM(
        model=args.model_path, 
        tensor_parallel_size=args.tensor_parallel_size, 
        trust_remote_code=True,
        max_model_len=args.max_model_len
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    stop_token_ids = get_stop_tokens(args.model_path, tokenizer)
    print(f"Using stop token IDs: {stop_token_ids}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids
    )

    print(f"Reading input file: {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    if 'question' not in df.columns:
        raise ValueError("Input CSV must contain a 'question' column.")

    total_samples = len(df)
    print(f"Processing {total_samples} samples in batches of {args.batch_size}...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    if os.path.exists(args.output_file):
        print(f"Overwriting existing file: {args.output_file}")
        os.remove(args.output_file)

    for i in range(0, total_samples, args.batch_size):
        batch_df = df.iloc[i : min(i + args.batch_size, total_samples)].copy()
        
        print(f"Batch {i//args.batch_size + 1}...")
        
        valid_prompts = []
        valid_indices = []
        
        for idx, row in batch_df.iterrows():
            # Check if chat template is available
            if tokenizer.chat_template is not None:
                messages = []
                if args.system_prompt:
                    messages.append({"role": "system", "content": args.system_prompt})
                messages.append({"role": "user", "content": str(row['question'])})
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback to simple instruction format
                # if args.system_prompt:
                #     prompt = f"{args.system_prompt}\n\nQuestion: {row['question']}\n\nAnswer:"
                # else:
                prompt = f"You are a helpful assistant. Solve the following query.\n\nQuestion: {row['question']}\n\nAnswer: "
            
            # Check length before adding to batch
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_token_ids) >= args.max_model_len:
                print(f"Skipping sample {idx}: Prompt too long ({len(prompt_token_ids)} tokens)")
                continue
                
            valid_prompts.append(prompt)
            valid_indices.append(idx)

        if not valid_prompts:
            print(f"No valid prompts in batch starting at {i}. Skipping.")
            continue

        outputs = llm.generate(valid_prompts, sampling_params)
        
        # Create a mapping for responses
        responses = {idx: output.outputs[0].text for idx, output in zip(valid_indices, outputs)}
        
        # Filter the batch_df to only include samples we actually generated for
        final_batch_df = batch_df.loc[valid_indices].copy()
        final_batch_df['response'] = [responses[idx] for idx in valid_indices]
        
        include_header = not os.path.exists(args.output_file)
        final_batch_df.to_csv(args.output_file, mode='a', index=False, header=include_header, quoting=csv.QUOTE_ALL, escapechar='\\')
        
        print(f"Saved {min(i + args.batch_size, total_samples)}/{total_samples}")

    print("Done.")

if __name__ == "__main__":
    main()
