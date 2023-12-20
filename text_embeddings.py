import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize  # You may need to install nltk by running: pip install nltk

def process_document(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_extension.lower() == '.pdf':
        return process_pdf_file(file_path)
    else:
        return None

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

def train_and_save_doc2vec_model(documents, model_path):
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(documents)]
    max_epochs = 100
    vec_size = 1536
    model = Doc2Vec(vector_size=vec_size, window=2, min_count=1, workers=4, epochs=max_epochs)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    model.save(model_path)

def load_and_infer_doc2vec_model(model_path, document_text):
    loaded_model = Doc2Vec.load(model_path)
    embedding = loaded_model.infer_vector(word_tokenize(document_text.lower()))
    return embedding

def main():
    input_directory = '/Users/anilaswani/Desktop/topic11/dataset2'
    model_path = 'doc2vec_model'

    documents = []
    for filename in os.listdir(input_directory):
        input_file_path = os.path.join(input_directory, filename)
        if 'txt' in input_file_path or 'pdf' in input_file_path:
            print(f"Processing {input_file_path}...")
            document_text = process_document(input_file_path)
            if document_text is not None:
                documents.append(document_text)

    train_and_save_doc2vec_model(documents, model_path)

    for i, filename in enumerate(os.listdir(input_directory)):
        input_file_path = os.path.join(input_directory, filename)

        document_text = process_document(input_file_path)

        if document_text is not None:
            embedding = load_and_infer_doc2vec_model(model_path, document_text)
            print(f"Embeddings for {filename}:\n{embedding}")

if __name__ == "__main__":
    main()
