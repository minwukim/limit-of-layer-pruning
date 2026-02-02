import os
os.environ["VLLM_USE_V1"] = "0"
import copy
import json
import shutil
import yaml
import importlib

from utils.merge_yaml_generator import generate_file
from utils.merge import merge_model, validate_config
from utils.evaluate import eval_math
from utils.prompts import QWEN_INSTRUCT

from transformers import AutoConfig

# --- Inputs ---
# Load configuration from YAML file
def load_config(config_path="input_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load inputs from YAML
input_config = load_config()

algo_iterations = input_config['algo_iterations']
if algo_iterations < 1: algo_iterations = 1

layers_to_remove = input_config['layers_to_remove']
layers_to_remove = sorted(layers_to_remove)
model_name = input_config['model_name']
eval_samples= input_config['eval_samples']
iterate = input_config['iterate']
merged_model_path = input_config['merged_model_path']
# Parse datasets - support comma-separated list
dataset_str = input_config['dataset']
datasets = [d.strip() for d in dataset_str.split(',')]

# Parse prompts - support comma-separated list to match datasets
prompts_module = importlib.import_module('utils.prompts')
prompt_names = input_config['prompt']
prompt_list = [p.strip() for p in prompt_names.split(',')]

# Create dataset->prompt mapping
if len(prompt_list) == 1:
    # Single prompt for all datasets
    dataset_prompts = {ds: getattr(prompts_module, prompt_list[0]) for ds in datasets}
else:
    # Each dataset gets its own prompt
    if len(prompt_list) != len(datasets):
        raise ValueError(f"Number of prompts ({len(prompt_list)}) must match number of datasets ({len(datasets)})")
    dataset_prompts = {ds: getattr(prompts_module, prompt_name) for ds, prompt_name in zip(datasets, prompt_list)}

config = AutoConfig.from_pretrained(model_name)
total_layers = config.num_hidden_layers

overwrite = input_config['overwrite']

def merge(model_name, layers_to_remove):
    generate_file(model_name, layers_to_remove)
    merge_model(output_path=merged_model_path)
    validate_config(output_path=merged_model_path)
    print("After ablation:", AutoConfig.from_pretrained(merged_model_path).num_hidden_layers, "layers")

def merge_and_eval(model_name, layers_to_remove, skip_merge=False):
    if not skip_merge: merge(model_name, layers_to_remove)

    print("Evaluating...")
    
    # Load the model once and reuse for all datasets
    from vllm import LLM
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    import torch
    
    model_path = merged_model_path if not skip_merge else model_name
    llm = LLM(
        trust_remote_code=True,
        model=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.9
    )
    
    results = {}
    for dataset in datasets:
        print(f"  Evaluating on dataset: {dataset} with prompt: {[k for k,v in dataset_prompts.items() if k == dataset]}")
        data = eval_math(model_path, 
                        dataset, 
                        eval_samples, 
                        dataset_prompts[dataset],  # Use dataset-specific prompt
                        llm=llm)  # Pass the LLM instance
        results[dataset] = {
            'final_accuracy': data['final_accuracy'],
            'items': data['items']
        }
    
    # Cleanup after all evaluations
    del llm
    cleanup_dist_env_and_memory()
    torch.cuda.empty_cache()
    
    return results

for it in range(algo_iterations):
    out_dir = "./outputs/" + model_name.replace("/", "_").replace(".", "_") + "__" + datasets[0] + ("_multi" if len(datasets) > 1 else "") + "__" + "-".join([str(i) for i in layers_to_remove])
    out_dir += "__iterate" if iterate else ""

    if os.path.exists(f"{out_dir}"):
        if overwrite: shutil.rmtree(out_dir); os.makedirs(out_dir)
    else: os.makedirs(f"{out_dir}")

    if not iterate:
        out_file = f"{out_dir}/results.json"
        if not overwrite and os.path.exists(out_file): print("Experiment already ran!"); exit(0)

        data = merge_and_eval(model_name, layers_to_remove)
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
    else:
        # 1. First, evaluate or load baseline to get normalization factors
        baseline_file = f"{out_dir}/baseline.json"
        baseline_accs = {}
        
        if os.path.exists(baseline_file):
            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)
                baseline_accs = {ds: baseline_data['datasets'][ds]['ablated_acc'] for ds in datasets}
                print(f"Loaded baseline accuracies: {baseline_accs}")
        else:
            print("Evaluating baseline (full model)...")
            data = merge_and_eval(model_name, layers_to_remove, skip_merge=True)
            baseline_accs = {ds: data[ds]['final_accuracy'] for ds in datasets}
            
            per_dataset_baseline = {}
            for ds_name in datasets:
                acc = data[ds_name]['final_accuracy']
                per_dataset_baseline[ds_name] = {
                    'baseline_acc': acc,
                    'ablated_acc': acc,
                    'normalized': 1.0,
                    'items': data[ds_name]['items']
                }
            
            baseline_output = {
                'layer': total_layers,
                'datasets': per_dataset_baseline,
                'avg_normalized': 1.0
            }
            with open(baseline_file, "w") as f:
                json.dump(baseline_output, f, indent=2)
            print(f"Baseline accuracies saved: {baseline_accs}")

        # 2. Now perform layer-by-layer ablation
        all_results = []  # List of results to pick the best layer from
        for i in range(total_layers): # 0 to total_layers-1
            all_results.append({'layer': i, 'per_dataset': {}, 'avg_normalized': -1})

            # Skip layer 0 (always preserve first layer) and already removed layers
            if i == 0 or i in layers_to_remove: continue
            
            out_file = f"{out_dir}/results_{i}.json"
            if not overwrite and os.path.exists(out_file):
                with open(out_file, "r") as f:
                    temp = json.load(f)
                    if isinstance(temp, dict) and 'datasets' in temp:
                        # Re-calculate normalized score if it was 1.0-defaulted in a previous bugged run
                        res_datasets = temp['datasets']
                        normalized_accs = []
                        for ds_name in datasets:
                            if ds_name in res_datasets:
                                ablated = res_datasets[ds_name]['ablated_acc']
                                base = baseline_accs.get(ds_name, 1.0)
                                norm = ablated / base if base > 0 else 0.0
                                res_datasets[ds_name]['normalized'] = norm
                                res_datasets[ds_name]['baseline_acc'] = base
                                normalized_accs.append(norm)
                        
                        avg_norm = sum(normalized_accs) / len(normalized_accs) if normalized_accs else 0.0
                        all_results[-1]['per_dataset'] = res_datasets
                        all_results[-1]['avg_normalized'] = avg_norm
                        
                        # Optionally update the file with corrected normalization
                        temp['avg_normalized'] = avg_norm
                        with open(out_file, "w") as f:
                            json.dump(temp, f, indent=2)
                    else:
                        # Fallback for very old format
                        all_results[-1]['avg_normalized'] = -1
                continue

            # Evaluate new ablation
            data = merge_and_eval(model_name, sorted(layers_to_remove + [i]))
            
            normalized_accs = []
            per_dataset_results = {}
            for ds_name in datasets:
                ablated_acc = data[ds_name]['final_accuracy']
                baseline_acc = baseline_accs.get(ds_name, 1.0)
                normalized = ablated_acc / baseline_acc if baseline_acc > 0 else 0.0
                
                normalized_accs.append(normalized)
                per_dataset_results[ds_name] = {
                    'baseline_acc': baseline_acc,
                    'ablated_acc': ablated_acc,
                    'normalized': normalized,
                    'items': data[ds_name]['items']
                }
            
            avg_normalized = sum(normalized_accs) / len(normalized_accs) if normalized_accs else 0.0
            all_results[-1]['per_dataset'] = per_dataset_results
            all_results[-1]['avg_normalized'] = avg_normalized
            
            output_data = {
                'layer': i,
                'datasets': per_dataset_results,
                'avg_normalized': avg_normalized
            }
            with open(out_file, "w") as f:
                json.dump(output_data, f, indent=2)

        # Select best layer based on highest average normalized accuracy
        # Filter out layer 0 (always preserve), already removed layers, and layers with invalid metrics
        valid_results = [(idx, res) for idx, res in enumerate(all_results) if idx != 0 and idx not in layers_to_remove and res['avg_normalized'] >= 0]
        
        if valid_results:
            sorted_results = sorted(
                valid_results,
                key=lambda x: (x[1]['avg_normalized'], x[0]),  # later index wins if normalized acc ties
                reverse=True
            )
            max_index = sorted_results[0][0]
            max_normalized = sorted_results[0][1]['avg_normalized']
            
            print(f"\nSelected layer {max_index} with avg normalized accuracy: {max_normalized:.4f}")
            print(f"Per-dataset normalized: {[(ds, all_results[max_index]['per_dataset'][ds]['normalized']) for ds in datasets]}")
            
            # add that layer in `layers_to_remove`
            layers_to_remove.append(max_index)
        else:
            print("No valid layers to remove!")
        
        # continue iteration again 

        print(f"""
        
        ALGO ITERATION {str(it)} DONE!!
        
        """)


print("""

ALL DONE!!!

""")


