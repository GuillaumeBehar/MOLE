from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from typing import Generator
import numpy as np


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

from models.rag import RAG
from models.llm import LLM
from utils.custom_utils import load_yaml
from utils.lecture_xml import get_data


class MetadataRAG(RAG):
    def __init__(self, llm: LLM, config: dict) -> None:
        """Initialize the class with the provided large language model."""
        super().__init__(llm=llm, config=config, name="MetadataRAG")
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2", device="cuda"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=64
        )

    def ingest(self, raw_documents: list[dict], show=False) -> None:
        """Ingest documents into the collection."""

        import time

        print("Ingesting documents...")
        now = time.time()
        documents = self.text_splitter.create_documents(
            [document["text"] for document in raw_documents],
            metadatas=[document["metadata"] for document in raw_documents
                       ],
        )
        if show:
            print("Text splitting time: ", time.time() - now)
            print("Number of chunk to ingest:", len(documents))
            print(
                "Mean length of documents to ingest:",
                np.mean([len(document["text"]) for document in raw_documents]),
            )
            print(
                "Mean length of chunks to ingest:",
                np.mean([len(doc.page_content) for doc in documents]),
            )

        current_count = self.collection.count()

        now = time.time()
        self.collection.add(
            documents=[document.page_content for document in documents],
            metadatas=[document.metadata for document in documents],
            ids=[str(k + current_count) for k in range(len(documents))],
        )
        if show:
            print("Ingestion time: ", time.time() - now)

    def retrieve(self, question: str) -> str:
        """Retrieves relevant context for the given question."""
        results = self.collection.query(query_texts=[question], n_results=5)
        return results

    def build_prompt(self, question: str, results: dict, data_getter=get_data) -> str:

        self.prompt_template = PromptTemplate.from_template(
            """
            As a specialized medical assistant for writing answers, your expertise lies in medicine. You may reference the provided sample document if necessary, but only if relevant. Each document is identified by its title, abstract, and retrieved chunks. If you use an information gived in a document, cite it with its ID in parentheses (e.g., (id)). Your primary goal is to provide accurate responses, adhering to a concise format of up to three sentences for clarity and efficiency. If unsure, simply acknowledge your inability to answer.
            Context : {context}
            Question : {question}
            Answer :
            """
        )

        self.context_template = PromptTemplate.from_template(
            """
            Document ID : {id}
            Title : {title}
            Abstract : {abstract}
            Chunks : {chunks}
            """
        )

        id_list = []
        metadata_list = []
        chunk_list = []
        for metadata_id in results["metadatas"][0]:
            id = metadata_id["id"]
            metadata = results["metadatas"][0][0]
            if id not in id_list:
                id_list.append(id)
                metadata_list.append(metadata)
                chunk_list += [[]]
                for i in range(len(results["metadatas"][0])):
                    if metadata["id"] == results["metadatas"][0][i]["id"]:
                        chunk_list[-1] += [results["documents"][0][i]]

        context = ""

        for metadata, chunks in zip(metadata_list, chunk_list):
            context += self.context_template.format(
                **{
                    "id": metadata["id"],
                    "title": metadata["title"],
                    "abstract": metadata["abstract"],
                    "chunks": "\n".join(chunks),
                }
            )

        prompt = self.prompt_template.format(
            **{"context": context, "question": question}
        )
        return prompt

    def build_prompt_yes_no(self, question: str, results: dict, data_getter) -> str:

        self.prompt_template = PromptTemplate.from_template(
            """
            As a specialized medical assistant for question-answering, your expertise lies in medicine. Each document is identified by its title, abstract, and retrieved chunks. Your goal is only to answer by yes or no. If unsure, simply acknowledge your inability to answer with maybe.
            Context : {context}
            Question : {question}
            Answer (Yes/No/Maybe) : 
            """
        )

        self.context_template = PromptTemplate.from_template(
            """
            Document ID : {id}
            Title : {title}
            Abstract : {abstract}
            Chunks : {chunks}
            """
        )

        id_list = []
        metadata_list = []
        chunk_list = []
        for metadata_id in results["metadatas"][0]:
            id = metadata_id["id"]
            metadata = results["metadatas"]
            if id not in id_list:
                id_list.append(id)
                metadata_list.append(metadata)
                chunk_list += [[]]
                for i in range(len(results["metadatas"][0])):
                    if metadata["id"] == results["metadatas"][0][i]["id"]:
                        chunk_list[-1] += [results["documents"][0][i]]

        context = ""

        for metadata, chunks in zip(metadata_list, chunk_list):
            context += self.context_template.format(
                **{
                    "id": metadata["id"],
                    "title": metadata["title"],
                    "abstract": metadata["abstract"],
                    "chunks": "\n".join(chunks),
                }
            )

        prompt = self.prompt_template.format(
            **{"context": context, "question": question}
        )
        return prompt

    def ask(
        self,
        question: str,
        web_search: bool = False,
    ) -> str:
        """Ask the RAG a question."""
        results = self.retrieve(question)
        prompt = self.build_prompt(question, results, get_data)
        return self.llm.ask(prompt)

    def ask_stream(
        self,
        question: str,
        web_search: bool = False,
    ) -> Generator:
        """Streams the response from the RAG."""
        results = self.retrieve(question)
        print(len(results["documents"][0]))
        prompt = self.build_prompt(question, results, get_data)
        print(prompt)
        return self.llm.ask_stream(prompt, web_search=web_search)


# Example usage of HugChatLLM
if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_yaml(MAIN_DIR_PATH + "./config.yaml")

    # Initialize the Large Language Model (LLM)
    llm = LLM(True, True, "llm")

    # Initialize the SimpleRAG instance with the LLM and configuration
    rag = MetadataRAG(llm, config)

    # Load the collection named "test4_collection"
    rag.load_collection("test_collection")

    # Ingest a batch of documents into the collection
    rag.ingest_batch(
        batch_size=8,
        doc_number=50,
        data_getter=get_data,
        doc_start=10500000,
        api=False,
        show=False,
    )

    # Ingest a list of documents into the collection
    # rag.ingest_list(
    #     batch_size=32,
    #     id_list=[10500064],
    #     data_getter=get_data,
    #     api=False,
    #     show=True,
    # )

    # Print the count of documents in the collection
    print("Number of chunk in the collection:", rag.collection.count())

    # Ask a question
    question = "How drought stress increases the proline accumulations in seedling leaves in woody plant species ?"
