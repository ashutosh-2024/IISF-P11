from bottle import Bottle, request
import json

from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize

from qdrant_client import QdrantClient
from qdrant_client.http import models

app = Bottle()

model_path = 'doc2vec_model'
doc2vec_model = Doc2Vec.load(model_path)

def qdrant_connection() -> QdrantClient:
    qdrant_client = QdrantClient(
        url="",
        api_key=""
    )
    return qdrant_client

with open ('mapper_results.json', 'r') as f:
    mapper_results = json.load(f)

def get_top_n_similar_documents(query_vector, n=3):
    client = qdrant_connection()
    return client.search(
                    collection_name='test_collection',
                    query_filter=models.Filter(),
                    search_params=models.SearchParams(
                        hnsw_ef=128,
                        exact=False
                    ),
                    query_vector = query_vector,
                    limit=n,
                    with_payload=True,
                )

def get_cluster_and_nearby_clusters(document_id):
    cluster = mapper_results.get(document_id, {}).get('cluster', None)
    nearby_clusters = set()
    for doc_id, info in mapper_results.items():
        if info.get('cluster') == cluster:
            nearby_clusters.add(doc_id)
    return cluster, nearby_clusters

def load_and_infer_doc2vec_model(model_path, document_text):
    loaded_model = Doc2Vec.load(model_path)
    embedding = loaded_model.infer_vector(word_tokenize(document_text.lower()))
    return embedding

@app.post('/get_results/<query_text>')
def get_results(query_text):
    try:
        query_vector = doc2vec_model.infer_vector(word_tokenize(query_text.lower()))
        all_docs = get_top_n_similar_documents([0.1] * 1536, n=1000)
        similar_documents = get_top_n_similar_documents(query_vector, n=3)
        new_all_docs = []
        new_similar_docs = []
        for doc in all_docs:
            new_all_docs = new_all_docs + doc.payload['file_name']
        for doc in similar_documents:
            new_similar_docs = new_similar_docs + doc.payload['file_name']
        indexes = []
        for document in new_similar_docs:
            indexes.append(new_all_docs.index(document))

        all_results = []
        for _,v in mapper_results.items():
            for index in indexes:
                if index in v:
                    all_results.extend(v)

        all_search_result_file_names = []
        for index in all_results:
            all_search_result_file_names.append(new_all_docs[index])
        return {'success': True, 'results': all_search_result_file_names}
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    app.run(host='localhost', port=8080)