import torch
import os
import numpy as np
from scipy.spatial.distance import cdist

# class TensorDistanceCalculator:
#     def __init__(self, tensor_dir):
#         self.tensor_dir = tensor_dir
#         self.tensors = self.load_tensors()

#     def load_tensors(self):
#         tensor_files = [f for f in os.listdir(self.tensor_dir) if f.endswith('.pt')]
#         tensors = [torch.load(os.path.join(self.tensor_dir, f))['last_layer_residual_stream'][-1].cpu().numpy() for f in tensor_files]
#         return tensors

#     def compute_distances(self, metric='cosine'):
#         distances = cdist([self.tensors[0]], self.tensors[1:], metric=metric)[0]
#         return distances

#     def save_distances_to_csv(self, distances, output_file="distances.csv"):
#         with open(output_file, "w") as f:
#             for distance in distances:
#                 f.write(f"{distance}\n")

# if __name__ == "__main__":
#     tensor_dir = "saved_tensors"
#     calculator = TensorDistanceCalculator(tensor_dir)
#     distances = calculator.compute_distances()
#     calculator.save_distances_to_csv(distances)



from sklearn.cluster import KMeans

class TensorClusterAssigner:
    def __init__(self, embedding_path, n_clusters=10):
        self.embedding_path = embedding_path
        self.n_clusters = n_clusters
        self.umap_embedding = self.load_umap_embedding()
        self.clusters = self.assign_clusters()

    def load_umap_embedding(self):
        embedding = torch.load(self.embedding_path).cpu().numpy()
        return embedding

    def assign_clusters(self):
        kmeans = KMeans(n_clusters=self.n_clusters)
        clusters = kmeans.fit_predict(self.umap_embedding)
        return clusters

    def save_clusters_to_csv(self, output_file="clusters2.csv"):
        with open(output_file, "w") as f:
            for cluster in self.clusters:
                f.write(f"{cluster}\n")

if __name__ == "__main__":
    embedding_path = "../../outputs/output_tensors/umap_embedding.pt"
    cluster_assigner = TensorClusterAssigner(embedding_path)
    cluster_assigner.save_clusters_to_csv()


# class EmbeddingDistanceCalculator:
#     def __init__(self, embedding_path):
#         self.embedding = torch.load(embedding_path).cpu().numpy()

#     def compute_distance(self, index1, index2, metric='cosine'):
#         embedding1 = self.embedding[int(index1)]  # Ensure index is an integer
#         embedding2 = self.embedding[int(index2)]  # Ensure index is an integer
#         distance = cdist([embedding1], [embedding2], metric=metric)[0][0]  # Compute distance
#         return distance

#     def print_distance(self, distance):
#         print("Distance:", distance)

# if __name__ == "__main__":
#     embedding_path = "output_tensors/umap_embedding.pt"
#     calculator = EmbeddingDistanceCalculator(embedding_path)

#     # Load saved tensors from saved_tensors directory
#     tensor_files = [f for f in os.listdir("saved_tensors") if f.endswith('.pt')]
#     tensors = [torch.load(os.path.join("saved_tensors", f))['last_layer_residual_stream'][-1].cpu().numpy() for f in tensor_files]

#     # Compute distances between the first vector and the remaining 997 vectors one by one and save to CSV
#     with open("distances.csv", "w") as f:
#         for i in range(1, len(tensors)):
#             distance = calculator.compute_distance(tensors[0], tensors[i], metric='cosine')
#             f.write(f"{distance}\n")