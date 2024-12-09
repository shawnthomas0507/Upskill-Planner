import chromadb
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings



def pdf_to_text(file_path):
    pdf_file=open(file_path,'rb')
    pdf_reader=PyPDF2.PdfReader(pdf_file)
    text=""
    for page_num in range( len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    pdf_file.close()
    return text


def insert_rag(file_path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    client = chromadb.PersistentClient(path="./db")

    collection_name = "my_collection"
    if collection_name in [col.name for col in client.list_collections()]:
        collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(name=collection_name)

    if file_path.endswith('.pdf'):
        text = pdf_to_text(file_path)
        chunks = text_splitter.split_text(text)

        documents_list = []
        embeddings_list = []
        ids_list = []

        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk)
            documents_list.append(chunk)
            embeddings_list.append(vector)
            ids_list.append(f"{os.path.basename(file_path)}_{i}")

        collection.add(
            embeddings=embeddings_list,
            documents=documents_list,
            ids=ids_list
        )

    return collection


