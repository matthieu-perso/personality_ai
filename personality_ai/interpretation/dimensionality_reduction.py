import torch
import umap
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class TensorVisualizer:
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.tensors = self.load_tensors()
        self.tensor_matrix = self.flatten_and_concatenate_tensors()
        self.tensor_matrix_np = self.tensor_matrix.cpu().numpy()

    def load_tensors(self):
        tensor_files = os.listdir(self.tensor_dir)
        tensors = []
        for f in tensor_files:
            file_path = os.path.join(self.tensor_dir, f)
            if os.path.getsize(file_path) > 0:  # Check if the file is not empty
                tensor = torch.load(file_path).get('common_embedding')
                if tensor is not None:
                    tensors.append(tensor)
                else:
                    print(f"Warning: 'common_embedding' not found in {file_path}")
            else:
                print(f"Warning: {file_path} is empty and will be skipped.")
        return tensors

    def flatten_and_concatenate_tensors(self):
        flattened_tensors = []
        for t in self.tensors:
            flattened_tensors.append(torch.flatten(t))
        tensor_matrix = torch.stack(flattened_tensors, dim=0)
        return tensor_matrix

    def run_umap(self, output_dir="output_tensors"):
        os.makedirs(output_dir, exist_ok=True)
        reducer = umap.UMAP(n_components=10)
        embedding = reducer.fit_transform(self.tensor_matrix_np)
        torch.save(torch.tensor(embedding), os.path.join(output_dir, "umap_embedding.pt"))
        return embedding

    def run_pca(self, output_dir="output_tensors"):
        os.makedirs(output_dir, exist_ok=True)
        pca = PCA(n_components=100)
        embedding = pca.fit_transform(self.tensors)
        torch.save(torch.tensor(embedding), os.path.join(output_dir, "pca_embedding.pt"))
        return embedding

    def visualize_embedding(self, embedding, title, save_path=None):
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, c=range(len(embedding)), cmap='Spectral')
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

if __name__ == "__main__":

    print("Running UMAP")

    tensor_dir = "saved_tensors"
    visualizer = TensorVisualizer(tensor_dir)

    umap_embedding = visualizer.run_umap()
    visualizer.visualize_embedding(umap_embedding, 'UMAP projection of the tensor data', save_path="umap_projection.png")

    pca_embedding = visualizer.run_pca()
    visualizer.visualize_embedding(pca_embedding, 'PCA projection of the tensor data', save_path="pca_projection.png")

