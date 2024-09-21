import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
from huggingface_hub import login


save_dir = "saved_tensors"
os.makedirs(save_dir, exist_ok=True)
# Login to Hugging Face Hub
login(token="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")

class Hook:
    def __init__(self):
        self.out = None
        self.attn = None  # Add this line to store attention weights

    def __call__(self, module, module_inputs, module_outputs):
        self.out, self.attn = module_outputs  # Modify this line to unpack attention weights

def load_llama(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model.to(device)
    return tokenizer, model

def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset.iloc[:, 0].tolist()
    return statements

def get_acts(statements, tokenizer, model, layers, device):
    """
    Get given layer activations and attention weights for the statements. 
    Return dictionary of stacked activations and attention weights.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    
    # get activations and attention weights
    acts = {layer : [] for layer in layers}
    attns = {layer : [] for layer in layers}
    for statement in tqdm(statements):
        input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
        model(input_ids)
        for layer, hook in zip(layers, hooks):
            acts[layer].append(hook.out[0, -1])
            attns[layer].append(hook.attn)
    
    for layer in layers:
        acts[layer] = t.stack(acts[layer]).float()
        attns[layer] = t.stack(attns[layer]).float()
    
    # remove hooks
    for handle in handles:
        handle.remove()
    
    return acts, attns

if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B",
                        help="Model name or path on Hugging Face model hub")
    parser.add_argument("--layers", nargs='+', 
                        help="Layers to save embeddings from")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="acts",
                        help="Directory to save activations to")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    t.set_grad_enabled(False)
    
    tokenizer, model = load_llama(args.model, args.device)
    for dataset in args.datasets:
        statements = load_statements(dataset)
        layers = [int(layer) for layer in args.layers]
        if layers == [-1]:
            layers = list(range(len(model.model.layers)))
        save_dir = f"{args.output_dir}/{args.model}/{dataset}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(0, len(statements), 25):
            acts = get_acts(statements[idx:idx + 25], tokenizer, model, layers, args.device)
            for layer, act in acts.items():
                    t.save(act, f"{save_dir}/layer_{layer}_{idx}.pt")



            