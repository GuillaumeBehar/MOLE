import chromadb
from typing import Generator
import xml.etree.ElementTree as ET
from tqdm import tqdm

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from utils.utils import *
from utils.lecture_xml import *

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
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def ingest(self, raw_documents) -> None:
        """Ingest documents into the collection."""
        pass  # This method will be implemented in subclasses

    def ingest_folder(self, folder_path: str) -> None:
        """Ingest every documents in a folder into the collection."""
        xml_paths = get_xml_paths(folder_path)
        for xml_path in tqdm(xml_paths, desc="Ingesting documents"):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            raw_documents = recup_abstract(root)
            self.ingest(raw_documents)

    def retrieve(self, question: str) -> str:
        """Retrieve relevant documents from the collection."""
        pass  # This method will be implemented in subclasses

    def ask(self, question: str) -> str:
        """Ask the RAG a question."""
        pass  # This method will be implemented in subclasses

    def ask_stream(self, prompt: str, web_search: bool = False) -> Generator:
        """Streams the response from the RAG."""
        pass  # This method will be implemented in subclasses
