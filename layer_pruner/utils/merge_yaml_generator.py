import yaml
from transformers import AutoConfig

def generate_file(model_name, layers_to_remove, out_fname="temp_slice.yaml"):
    
    config = AutoConfig.from_pretrained(model_name)
    total_layers = config.num_hidden_layers
    
    # --- Compute layer ranges to keep ---
    ranges = []
    start = 0
    for layer in layers_to_remove:
        if start < layer:
            ranges.append([start, layer])
        start = layer + 1
    if start <= total_layers:
        ranges.append([start, total_layers])

    # --- Build YAML structure ---
    slices = []
    for r in ranges:
        slices.append({
            "sources": [{
                "model": model_name,
                "layer_range": r
            }]
        })

    yaml_data = {
        "slices": slices,
        "merge_method": "passthrough",
        "dtype": str(config.torch_dtype).split(".")[-1]
    }

    # --- Custom Representer for inline layer_range only ---
    class InlineLayerRangeDumper(yaml.Dumper):
        pass

    def represent_inline_layer_range(dumper, data):
        # Only make [x, y] inline for layer_range lists
        if len(data) == 2 and all(isinstance(i, int) for i in data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        else:
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data)

    InlineLayerRangeDumper.add_representer(list, represent_inline_layer_range)

    # --- Write YAML ---
    with open(out_fname, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False, Dumper=InlineLayerRangeDumper, indent=2)

    print(yaml.dump(yaml_data, sort_keys=False, Dumper=InlineLayerRangeDumper, indent=2))