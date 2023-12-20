import kmapper as km
import torch
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, Birch, OPTICS 
import img2vec_pytorch as i2v
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import uuid
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from umap import UMAP

def qdrant_connection() -> QdrantClient:
    qdrant_client = QdrantClient(
        url="url",
        api_key="apikey",
    )
    return qdrant_client

def get_image_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    vec = img2vec.get_vec(img)
    return vec.tolist()

def build_index(root_dir, qdrant_client):
    index = []
    for dir_name, _, file_list in os.walk(root_dir):
        for file_name in file_list:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dir_name, file_name)
                embedding = get_image_embedding(img_path)

                index.append(embedding)
                print('---------------')
                print(len(embedding))

                qdrant_client.upsert(
                    collection_name="bhuvan",
                    points=models.Batch(
                        ids=[uuid.uuid4().hex],
                        payloads=[{"dir_name": dir_name,"file_name": file_name,}],
                        vectors=[embedding],
                    ),
                )
    return index

def reduce_dimensionality(embedding, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_embedding = pca.fit_transform(np.array(embedding))
    return reduced_embedding.tolist()

def reduce_dimensionality_tsne(embedding, n_components=2):
    tsne = TSNE(n_components=n_components)
    reduced_embedding = tsne.fit_transform(np.array(embedding))
    return reduced_embedding.tolist()

def reduce_dimensionality_umap(embedding, n_components=2):
    umap = UMAP(n_components=n_components)
    embedding_array = np.array(embedding)
    reduced_embedding = umap.fit_transform(embedding_array)
    return reduced_embedding.tolist()

def cluster_metrics(true_labels, predicted_labels):
    ri = metrics.rand_score(true_labels, predicted_labels)
    ari = metrics.adjusted_rand_score(true_labels, predicted_labels)
    nmi = metrics.normalized_mutual_info_score(true_labels, predicted_labels)
    fm_index = metrics.fowlkes_mallows_score(true_labels, predicted_labels)
    return [ri, ari, nmi, fm_index]

root_directory = "topic11/d1/Training Data"
img2vec = i2v.Img2Vec(model='efficientnet_b3')

qdrant_client = qdrant_connection()
index = qdrant_client.search(
    collection_name="bhuvan",
    query_vector=[0.1] * 1536,
    limit=1000,
    with_vectors=True,
    with_payload=True,
)
labels = []
for k in range(len(index)):
    payload = (index[k]).payload
    directory = payload['dir_name']
    folder = directory.split('/')[-1]
    labels.append(folder)
    print(k, folder)

index_new = []
for i in index:
    index_new.append(i.vector)

dimensions = 5

mapper = km.KeplerMapper(verbose=1)

kmeans_clusterer = KMeans(n_clusters=1, n_init='auto')
agglomerative_clusterer = AgglomerativeClustering(n_clusters=2)
dbscan_clusterer = DBSCAN()
spectral_clusterer = SpectralClustering(n_clusters=2)
birch_clusterer = Birch(n_clusters=2)
optics_clusterer = OPTICS()


# Generate Metrics
dimensionality_reduction = ["umap", "pca", "tsne"]
clustering = [
    "kmeans", 
    "agglomerative", 
    "dbscan", 
    "spectral", 
    # "birch", 
    "optics"
    ]
for d in dimensionality_reduction:
    projected_data = None
    if d == "pca":
        index = reduce_dimensionality((index_new), n_components=dimensions)
        projected_data = mapper.fit_transform(np.array(index), projection=list(range(dimensions)))
    elif d == "tsne":
        index = reduce_dimensionality_tsne(np.array(index_new), n_components=dimensions)
        projected_data = mapper.fit_transform(np.array(index), projection=list(range(dimensions)))
    elif d == "umap":
        index = reduce_dimensionality_umap(np.array(index_new), n_components=dimensions)
        projected_data = mapper.fit_transform(np.array(index), projection=list(range(dimensions)))
    
    for c in clustering:
        try:
            if c == "kmeans":
                graph = mapper.map(projected_data, np.array(index), clusterer=kmeans_clusterer)
            elif c == "agglomerative":
                graph = mapper.map(projected_data, np.array(index), clusterer=agglomerative_clusterer)
            elif c == "dbscan":
                graph = mapper.map(projected_data, np.array(index), clusterer=dbscan_clusterer)
            elif c == "spectral":
                graph = mapper.map(projected_data, np.array(index), clusterer=spectral_clusterer)
            elif c == "birch":
                graph = mapper.map(projected_data, np.array(index), clusterer=birch_clusterer)
            elif c == "optics":
                graph = mapper.map(projected_data, np.array(index), clusterer=optics_clusterer)
            
            true_labels = labels  
            predicted_labels = []
            done = []
            for cluster_id, cluster_info in enumerate(graph['nodes']):
                cluster_indices = graph["nodes"][cluster_info]
                for i in cluster_indices:
                    if i not in done:
                        predicted_labels.append(labels[i])
                        done.append(i)

            evaluation_scores = cluster_metrics(true_labels, predicted_labels)

            print(f"For {d} with {c}")
            print("Rand Index:", evaluation_scores[0])
            print("Adjusted Rand Index:", evaluation_scores[1])
            print("Normalized Mutual Info:", evaluation_scores[2])
            print("Fowlkes-Mallows Index:", evaluation_scores[3])
        
        except Exception as e:
            print (f"error on {d} with {c}", e)
