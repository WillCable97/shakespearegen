import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming you have trained an RNN model and obtained word embeddings
# Replace this with your actual trained RNN model and data
# For demonstration purposes, let's generate random embeddings and labels
num_words = 1000
embedding_dim = 100
embeddings = np.random.rand(num_words, embedding_dim)

# Generate random labels for each embedding
labels = [str(i) for i in range(num_words)]

# Perform t-SNE dimensionality reduction on embeddings
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

# Perform k-means clustering on the embeddings
num_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings_tsne)

# Define colors for each cluster
cluster_colors = ['blue', 'green', 'red', 'purple', 'orange']  # Add more colors as needed

# Plot t-SNE visualization with annotated closest examples per cluster and background colors
plt.figure(figsize=(10, 8))
for i in range(num_clusters):
    cluster_points = embeddings_tsne[cluster_labels == i][:5]  # Limit to 5 examples per cluster
    centroid = kmeans.cluster_centers_[i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], marker='o', color=cluster_colors[i], label=f'Cluster {i}')
    plt.scatter(centroid[0], centroid[1], marker='x', color='black', s=100)
    
    # Annotate closest embeddings with labels
    for point in cluster_points:
        idx = np.where((embeddings_tsne == point).all(axis=1))[0][0]
        plt.annotate(labels[idx], (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('t-SNE Visualization of Word Embeddings with Clusters and Closest Examples')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()
