import torch
import umap
import shap
import numpy as np
import glob
import os

# Load UMAP embeddings
embeddings = torch.load('../outputs/output_tensors/umap_embedding.pt').numpy()

# Load original high-dimensional data
tensor_files = glob.glob(os.path.join(os.getcwd(), '../saved_tensors', '*.pt'))

if not tensor_files:
    print("No tensor files found. Please check the path.")
else:
    original_data_list = []
    for f in tensor_files:
        try:
            tensor_dict = torch.load(f)
            if 'common_embedding' in tensor_dict:
                original_data_list.append(tensor_dict['common_embedding'].cpu().numpy())
            else:
                print(f"'common_embedding' not found in {f}")
        except EOFError:
            print(f"EOFError: {f} is empty or corrupted.")
        except Exception as e:
            print(f"An error occurred while loading {f}: {e}")

    if original_data_list:
        #original_data = np.concatenate(original_data_list, axis=0)
        #print(f"Original data shape: {original_data.shape}")
        
        # Fit UMAP model
        umap_model = umap.UMAP()
        umap_model.fit(original_data_list)

        # Use SHAP to explain the UMAP model
        explainer = shap.Explainer(umap_model)
        shap_values = explainer(original_data_list)

        # Plot the SHAP values
        shap.summary_plot(shap_values, original_data_list)
    else:
        print("No valid common embeddings found.")