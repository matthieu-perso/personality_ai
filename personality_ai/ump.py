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
        tensor_files = [f for f in os.listdir(self.tensor_dir) if f.endswith('.pt')]
        tensors = [torch.load(os.path.join(self.tensor_dir, f))['common_embedding'] for f in tensor_files]
        return tensors

    def flatten_and_concatenate_tensors(self):
        flattened_tensors = []
        for t in self.tensors:
            # print(type(t))
            # print(t.keys())
            # print(t)
            print(t['last_layer_residual_stream'].shape)
            flattened_tensors.append(torch.flatten(t['last_layer_residual_stream'][-1]))
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
        embedding = pca.fit_transform(self.tensor_matrix_np)
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
    tensor_dir = "saved_tensors"
    visualizer = TensorVisualizer(tensor_dir)
    print("Running UMAP")

    umap_embedding = visualizer.run_umap()
    visualizer.visualize_embedding(umap_embedding, 'UMAP projection of the tensor data', save_path="umap_projection.png")

    pca_embedding = visualizer.run_pca()
    visualizer.visualize_embedding(pca_embedding, 'PCA projection of the tensor data', save_path="pca_projection.png")

