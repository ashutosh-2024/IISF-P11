import os
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

def process_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_extension.lower() == '.pdf':
        return process_pdf_file(file_path)
    else:
        return None
    

def qdrant_connection() -> QdrantClient:
    qdrant_client = QdrantClient(
        url="",  # URL of the Qdrant instance
        api_key="",  # API key of the Qdrant instance
    )
    return qdrant_client

def create_index():

    client = qdrant_connection()

    client.create_collection(
    collection_name="text_collection",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
)
    
def process_pdf_file(file_path):
    pdf_text = ""
    try:
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(file_path)
        length = len(pdf_reader.pages)
        print(f"Total number of pages in the PDF document: {length}")
        for page_num in range(length):
            text = pdf_reader.pages[page_num].extract_text()
            pdf_text += text
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
    return pdf_text

def load_and_infer_doc2vec_model(model_path, document_text):
    loaded_model = Doc2Vec.load(model_path)
    embedding = loaded_model.infer_vector(word_tokenize(document_text.lower()))
    return embedding

def main():

    # create_index()
    input_directory = '/Users/anilaswani/Desktop/topic11/dataset2'
    model_path = 'doc2vec_model'

    for i, filename in enumerate(os.listdir(input_directory)):
        input_file_path = os.path.join(input_directory, filename)

        document_text = process_document(input_file_path)

        if document_text is not None:
            embedding = load_and_infer_doc2vec_model(model_path, document_text)
            client = qdrant_connection()
            client.upsert(
                collection_name="text_collection",
                points=models.Batch(
                    ids=[uuid.uuid4().hex],
                    payloads=[{"file_name": filename,}],
                    vectors=[embedding],
                ),
            )

if __name__ == "__main__":
    main()
