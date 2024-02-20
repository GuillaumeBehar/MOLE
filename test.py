from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from models.rag import *

MAIN_DIR_PATH = up(os.path.abspath(__file__))

config = load_yaml(MAIN_DIR_PATH + "./config.yaml")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)

langchain_documents = [
    {
        "metadata": {
            "id": 1,
            "title": "Document 1",
            "author": "John Doe",
            "date": "2022-02-15",
            # Add any other metadata fields you need
        },
        "text": "This is the content of document 1. It may span multiple sentences or paragraphs.",
    },
    {
        "metadata": {
            "id": 2,
            "title": "Document 2",
            "author": "Jane Smith",
            "date": "2022-02-16",
            # Add any other metadata fields you need
        },
        "text": "This is the content of document 2. It may also contain multiple sentences or paragraphs.",
    },
    # Add more documents as needed
]

# Split each document's text content using the splitter
split_documents = []
for document in langchain_documents:
    metadata = document["metadata"]
    text = document["text"]
    split_text = text_splitter.split_text(text)
    split_documents.extend(
        [{"metadata": metadata, "text": chunk} for chunk in split_text]
    )

print(split_documents)

# Assuming you have initialized your ChromaDB client and collection
chroma_client = chromadb.PersistentClient(
    MAIN_DIR_PATH + config["collections_directory"]
)
collection_name = "test4_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Iterate over each split document and ingest it into the collection
for split_document in split_documents:
    metadata = split_document["metadata"]
    text = split_document["text"]

    # Create a ChromaDB Document object with the metadata and text
    collection.add(documents=[text], metadatas=[metadata], ids=[str(metadata["id"])])
print(collection.count())
