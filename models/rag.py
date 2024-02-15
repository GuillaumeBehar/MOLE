import chromadb
from typing import Generator

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

MAIN_DIR_PATH = up(up(os.path.abspath(__file__)))


class RAG:
    def __init__(self, llm, config, name) -> None:
        """Initialize the RAG class with the provided large language model."""
        self.name = name
        self.llm = llm
        self.config = config
        self.embedding_function = None
        self.text_splitter = None
        self.prompt_template = None
        self.chroma_client = chromadb.PersistentClient(
            MAIN_DIR_PATH + config["collections_directory"]
        )

    def load_collection(self, collection_name: str) -> None:
        """Load or create a collection for retrieval."""
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except Exception as e:
            print("Error loading collection: %s", e)
            self.collection = self.chroma_client.create_collection(name=collection_name)

    def ingest(self, raw_documents) -> None:
        """Ingest documents into the collection."""
        pass  # This method will be implemented in subclasses

    def ingest_folder(self, raw_documents: list[str]) -> None:
        """Ingest documents into the collection."""
        documents = self.text_splitter.create_documents(raw_documents)
        self.collection.add(
            documents=[documents[k].page_content for k in range(len(documents))],
            ids=[str(k) for k in range(len(documents))],
        )

    def retrieve(self, question: str) -> str:
        """Retrieve relevant documents from the collection."""
        pass  # This method will be implemented in subclasses

    def ask(self, question: str) -> str:
        """Ask the RAG a question."""
        pass  # This method will be implemented in subclasses

    def ask_stream(self, prompt: str, web_search: bool = False) -> Generator:
        """Streams the response from the RAG."""
        pass  # This method will be implemented in subclasses
