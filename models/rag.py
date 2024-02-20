import chromadb
from typing import Generator
import xml.etree.ElementTree as ET
from tqdm import tqdm

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from models.hugchat_llm import *
from models.local_llm import *
from utils.utils import *
from utils.lecture_xml import *

MAIN_DIR_PATH = up(up(os.path.abspath(__file__)))


class RAG:
    def __init__(self, llm: LLM, config: dict, name: str) -> None:
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
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def ingest(self, raw_documents) -> None:
        """Ingest documents into the collection."""
        pass  # This method will be implemented in subclasses

    def ingest_batch(
        self, batch_size: int, doc_number: int, data_getter, doc_start: int, api: bool
    ) -> None:
        """Ingest a batch of documents into the collection."""
        index = doc_start
        doc_gotten = 0

        import time

        for _ in tqdm(range(doc_number // batch_size), desc="Ingesting documents"):
            now = time.time()
            raw_documents = []
            for k in range(batch_size):
                try:
                    document = data_getter(index + k, api)
                    if document is not None:
                        raw_documents.append(document)
                except Exception as e:
                    # print(f"Error retrieving document {index + k}: {e}")
                    pass
            print(
                f"Retrieved {len(raw_documents)} documents in {time.time() - now} seconds"
            )
            now = time.time()
            if len(raw_documents) > 0:
                self.ingest(raw_documents)
                print(
                    f"Ingested {len(raw_documents)} documents in {time.time() - now} seconds"
                )
            index += batch_size
            doc_gotten += batch_size

    def retrieve(self, question: str) -> str:
        """Retrieve relevant documents from the collection."""
        pass  # This method will be implemented in subclasses

    def ask(self, question: str) -> str:
        """Ask the RAG a question."""
        pass  # This method will be implemented in subclasses

    def ask_stream(self, prompt: str, web_search: bool = False) -> Generator:
        """Streams the response from the RAG."""
        pass  # This method will be implemented in subclasses
