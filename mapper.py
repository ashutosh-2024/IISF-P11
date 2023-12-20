import kmapper as km
import torch
from sklearn.datasets import make_blobs
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
        url="",
        api_key="",
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
    reduced_embedding = pca.fit_transform(embedding)
    return reduced_embedding.tolist()

def reduce_dimensionality_tsne(embedding, n_components=2):
    tsne = TSNE(n_components=n_components)
    reduced_embedding = tsne.fit_transform(embedding)
    return reduced_embedding.tolist()

def reduce_dimensionality_umap(embedding, n_components=2):
    umap = UMAP(n_components=n_components)
    embedding_array = np.array(embedding)
    reduced_embedding = umap.fit_transform(embedding_array)
    return reduced_embedding.tolist()

root_directory = "/Users/anilaswani/Downloads/topic11/Sample Dataset1/Training Data"
img2vec = i2v.Img2Vec(model='efficientnet_b3')

qdrant_client = qdrant_connection()
index = qdrant_client.search(
    collection_name="bhuvan",
    query_vector=[0.1] * 1536,
    limit=1000,
    with_vectors=True,
    with_payload=True,
)

# for k in range(len(index)):
#     payload = (index[k]).payload
#     directory = payload['dir_name']
#     folder = directory.split('/')[-1]
#     print(k, folder)

# index_new = []
# for i in index:
#     index_new.append(i.vector)

dimensions = 5
# index = reduce_dimensionality((index_new), n_components=dimensions)
# index = reduce_dimensionality_tsne(np.array(index_new), n_components=dimensions)
index = reduce_dimensionality_umap(np.array(index_new), n_components=dimensions)

mapper = km.KeplerMapper(verbose=1)
projected_data = mapper.fit_transform(np.array(index), projection=list(range(dimensions)))

kmeans_clusterer = KMeans(n_clusters=1)
agglomerative_clusterer = AgglomerativeClustering(n_clusters=2)
dbscan_clusterer = DBSCAN()
spectral_clusterer = SpectralClustering(n_clusters=2)
birch_clusterer = Birch(n_clusters=2)
optics_clusterer = OPTICS()

kmeans_graph = mapper.map(projected_data, np.array(index), clusterer=kmeans_clusterer)
agglomerative_graph = mapper.map(projected_data, np.array(index), clusterer=agglomerative_clusterer)
dbscan_graph = mapper.map(projected_data, np.array(index), clusterer=dbscan_clusterer)
spectral_graph = mapper.map(projected_data, np.array(index), clusterer=spectral_clusterer)
birch_graph = mapper.map(projected_data, np.array(index), clusterer=birch_clusterer)
optics_graph = mapper.map(projected_data, np.array(index), clusterer=optics_clusterer)

mapper.visualize(kmeans_graph, path_html="rooms-KMeans.html", title="Rooms")
mapper.visualize(agglomerative_graph, path_html="rooms-Agglomerative.html", title="Rooms")
# mapper.visualize(dbscan_graph, path_html="rooms-DBSCAN.html", title="Rooms")
mapper.visualize(spectral_graph, path_html="rooms-Spectral.html", title="Rooms")
mapper.visualize(birch_graph, path_html="rooms-Birch.html", title="Rooms")
mapper.visualize(optics_graph, path_html="rooms-OPTICS.html", title="Rooms")
