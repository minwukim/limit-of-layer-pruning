import torch
import yaml
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
import os
import shutil
import json

def merge_model(output_path="./merged", config_yml="./temp_slice.yaml"):
    OUTPUT_PATH = output_path  # folder to store the result in
    LORA_MERGE_CACHE = "/tmp"  # change if you want to keep these for some reason
    CONFIG_YML = config_yml  # merge configuration file
    COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
    LAZY_UNPICKLE = False  # experimental low-memory model loader
    LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap
    
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(output_path)
        print("Deleted current model.")
        
    # actually do merge
    with open(CONFIG_YML, "r", encoding="utf-8") as fp:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

    run_merge(
        merge_config,
        out_path=OUTPUT_PATH,
        options=MergeOptions(
            lora_merge_cache=LORA_MERGE_CACHE,
            cuda=torch.cuda.is_available(),
            copy_tokenizer=COPY_TOKENIZER,
            lazy_unpickle=LAZY_UNPICKLE,
            low_cpu_memory=LOW_CPU_MEMORY,
        ),
    )
    print("Done!")

def validate_config(output_path="./merged"):
    with open(f"{output_path}/config.json", "r") as f:
        config = json.load(f)

        if 'num_hidden_layers' in config.keys() and 'layer_types' in config.keys():
            if config['num_hidden_layers'] != len(config['layer_types']):
                config['layer_types'] = config['layer_types'][:config['num_hidden_layers']]
    
    with open(f"{output_path}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Successfully validated config!")