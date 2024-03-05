from time import perf_counter
import chromadb
from typing import Generator
from tqdm import tqdm


def get_main_dir(depth: int = 0):  # nopep8
    """Get the main directory of the project."""
    import os
    import sys
    from os.path import dirname as up
    main_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(depth):
        sys.path.append(up(main_dir))
        main_dir = up(main_dir)
    return main_dir


MAIN_DIR_PATH = get_main_dir(1)  # nopep8


from utils.lecture_xml import *
from utils.custom_utils import *
from models.local_gguf_llm import *
from models.hugchat_llm import *


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

    def ingest_batch(self,
                     batch_size: int,
                     doc_number: int,
                     doc_start: int,
                     data_getter, api: bool,
                     show: bool = False) -> None:
        """Ingest a batch of documents into the collection."""
        index = doc_start
        doc_ingested = 0

        for _ in tqdm(range(doc_number // batch_size), desc="Ingesting documents"):
            start_time = perf_counter()
            raw_documents = []
            while len(raw_documents) != batch_size or doc_number <= doc_ingested:
                document = data_getter(index, api, show)
                index += 1
                if document is not None:
                    raw_documents.append(document)
                    doc_ingested += 1
            if show:
                print(
                    f"Retrieved {len(raw_documents)} documents in {perf_counter() - start_time:.2f} seconds")
            start_time = perf_counter()
            if raw_documents:
                self.ingest(raw_documents)
                if show:
                    print(
                        f"Ingested {len(raw_documents)} documents in {perf_counter() - start_time:.2f} seconds")
            index += batch_size

    def ingest_list(self,
                    batch_size: int,
                    id_list: list[int],
                    data_getter, api: bool,
                    show: bool = False) -> None:
        """Ingest a batch of documents into the collection."""

        for ids in tqdm(group_list(id_list, batch_size), desc="Ingesting documents"):
            start_time = perf_counter()
            raw_documents = [document for document in (data_getter(
                id, api, show) for id in ids) if document is not None]
            if show:
                print(
                    f"Retrieved {len(raw_documents)} documents in {perf_counter() - start_time:.2f} seconds")
            start_time = perf_counter()
            if raw_documents:
                self.ingest(raw_documents)
                if show:
                    print(
                        f"Ingested {len(raw_documents)} documents in {perf_counter() - start_time:.2f} seconds")

    def retrieve(self, question: str) -> str:
        """Retrieve relevant documents from the collection."""
        pass  # This method will be implemented in subclasses

    def ask(self, question: str) -> str:
        """Ask the RAG a question."""
        pass  # This method will be implemented in subclasses

    def ask_stream(self, prompt: str, web_search: bool = False) -> Generator:
        """Streams the response from the RAG."""
        pass  # This method will be implemented in subclasses
