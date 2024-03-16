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
num_clusters = 5  # Adjust the number of clusters as needed
num_per_cluster = -1
labels = [str(i) for i in range(num_words)]

def create_word_vis(emeddings, labels, num_clusters, per_cluster, show_labels = False, show_centroid= True):
    # Perform t-SNE dimensionality reduction on embeddings
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Perform k-means clustering on the embeddings

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_tsne)

    # Define colors for each cluster
    cluster_colors = ['blue', 'green', 'red', 'purple', 'orange']  # Add more colors as needed

    # Plot t-SNE visualization with annotated closest examples per cluster and background colors
    plt.figure(figsize=(10, 8))
    for i in range(num_clusters):
        cluster_points = embeddings_tsne[cluster_labels == i][:per_cluster]  # Limit to 5 examples per cluster
        centroid = kmeans.cluster_centers_[i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], marker='o', color=cluster_colors[i], label=f'Cluster {i}')
        if show_centroid: plt.scatter(centroid[0], centroid[1], marker='x', color='black', s=100)
        
        # Annotate closest embeddings with labels
        for point in cluster_points:
            idx = np.where((embeddings_tsne == point).all(axis=1))[0][0]
            if show_labels: plt.annotate(labels[idx], (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('t-SNE Visualization of Word Embeddings with Clusters and Closest Examples')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()

import sys
import os
import tensorflow as tf

root_path = os.path.abspath("./")
sys.path.append(root_path)


from src.models.RecurrentModels.RNN_model import RNN_model

model_for_testing = RNN_model(66, 256, 512, 1)
model_for_testing.build(tf.TensorShape([1, None]))


model_weight_path = os.path.join(root_path, "models", "RNN100Seq256Emb512Dense", "checkpoint_tracker", "ckpt_1.weights.h5")



model_for_testing.load_weights(model_weight_path)


embedding_layer = model_for_testing.layers[0]
embedding_weights = embedding_layer.get_weights()[0]
vocab = embedding_weights.shape[0]


from src.data.TextToToken.CustomCharacterToken import CustomCharacterToken

my_token = CustomCharacterToken(use_bookmark=False)
tokens = [str(i) for i in range(vocab)]
token_vals = [my_token.detokenise([[x]]) for x in tokens]


create_word_vis(embedding_weights, token_vals, 5, -1)