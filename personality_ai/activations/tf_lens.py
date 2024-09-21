import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
from huggingface_hub import login
from transformer_lens import HookedTransformer

save_dir = "saved_tensors"
os.makedirs(save_dir, exist_ok=True)
# Login to Hugging Face Hub
login(token="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")

def load_dataset_and_set_texts(dataset_name):
    """
    Load the dataset from the datasets folder and set the gpt2_texts as a list of descriptions from the dataset.
    
    Args:
    dataset_name (str): The name of the dataset file (without extension).
    
    Returns:
    list: A list of descriptions from the dataset.
    """
    dataset_path = f"datasets/{dataset_name}.csv"
    dataset = pd.read_csv(dataset_path)
    gpt2_texts = dataset.iloc[:, 0].tolist()
    return gpt2_texts

def get_attention_based_embedding(residual_stream, attention_weights):
    """
    Function to get the attention-based embedding from the residual stream.
    
    Args:
    residual_stream (torch.Tensor): The residual stream tensor from which to extract the embedding.
    attention_weights (torch.Tensor): The attention weights tensor.
    
    Returns:
    torch.Tensor: The attention-based embedding.
    """
    # Average attention weights over all heads
    attention_weights = attention_weights.mean(dim=0)
    
    # Normalize attention weights using softmax
    attention_weights = t.nn.functional.softmax(attention_weights, dim=-1)
    
    # Compute the weighted sum of hidden states
    attention_based_embedding = t.matmul(attention_weights, residual_stream)
    return attention_based_embedding

def get_common_embedding(attention_based_embedding):
    """
    Function to get a common embedding for the paragraph using mean pooling.
    
    Args:
    attention_based_embedding (torch.Tensor): The attention-based embedding tensor.
    
    Returns:
    torch.Tensor: The common embedding for the paragraph.
    """
    # Mean pooling
    common_embedding = attention_based_embedding.mean(dim=0)
    return common_embedding

dataset_name = "character_descriptions"
gpt2_texts = load_dataset_and_set_texts(dataset_name)

hooked_model = HookedTransformer.from_pretrained("phi-3")

for idx, text in enumerate(gpt2_texts):
    try:
        tokens = hooked_model.to_tokens(text)
        logits, cache = hooked_model.run_with_cache(tokens, remove_batch_dim=True)
        
        # Extract the residual stream for the last layer
        last_layer_residual_stream = cache["resid_post", -1]  # -1 to get the last layer
        
        # Extract the attention weights for the last layer
        attention_weights = cache["attn_scores", -1]  # -1 to get the last layer
        
        # Get the attention-based embedding
        attention_based_embedding = get_attention_based_embedding(last_layer_residual_stream, attention_weights)
        
        # Get the common embedding for the paragraph
        common_embedding = get_common_embedding(attention_based_embedding)

        print(f"Shape of common embedding at index {idx}: {common_embedding.shape}")
        
        # Create a dictionary of tensors
        tensor_dict = {
            "tokens": tokens,
            "logits": logits,
            "last_layer_residual_stream": last_layer_residual_stream,
            "attention_based_embedding": attention_based_embedding,
            "common_embedding": common_embedding
        }
        
        # Save the dictionary
        t.save(tensor_dict, os.path.join(save_dir, f"tensors_{idx}.pt"))
        
    except Exception as e:
        print(f"An error occurred at index {idx} with text: {text}")
        print(f"Error: {e}")
        # Optionally, log the error to a file for further analysis
        with open(os.path.join(save_dir, "error_log.txt"), "a") as log_file:
            log_file.write(f"An error occurred at index {idx} with text: {text}\n")
            log_file.write(f"Error: {e}\n")