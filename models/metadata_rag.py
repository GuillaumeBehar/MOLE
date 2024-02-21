from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from models.rag import *


class MetadataRAG(RAG):
    def __init__(self, llm: LLM, config: dict) -> None:
        """Initialize the class with the provided large language model."""
        super().__init__(llm=llm, config=config, name="SimpleRetrieveRAG")
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2", device="cuda"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=64
        )

    def ingest(self, raw_documents: list[dict]) -> None:
        """Ingest documents into the collection."""

        documents = self.text_splitter.create_documents(
            [document["text"] for document in raw_documents],
            metadatas=[document["metadata"] for document in raw_documents],
        )

        current_count = self.collection.count()

        self.collection.add(
            documents=[document.page_content for document in documents],
            metadatas=[document.metadata for document in documents],
            ids=[str(k + current_count) for k in range(len(documents))],
        )

    def retrieve(self, question: str) -> str:
        """Retrieves relevant context for the given question."""
        results = self.collection.query(query_texts=[question], n_results=5)
        return results

    def build_prompt(self, question: str, results: dict) -> str:

        self.prompt_template = PromptTemplate.from_template(
            """
            Context : {context}
            As a specialized medical assistant for question-answering, your expertise lies in medicine. You may reference the provided sample document if necessary, but only if relevant. Each document is identified by its title, abstract, and retrieved chunks. If you use an information gived in a document, cite it with its ID in parentheses (e.g., (id)). Your primary goal is to provide accurate responses, adhering to a concise format of up to three sentences for clarity and efficiency. If unsure, simply acknowledge your inability to answer.
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
        for metadata in results["metadatas"][0]:
            if metadata["id"] not in id_list:
                id_list.append(metadata["id"])
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

    def build_prompt_yes_no(self, question: str, results: dict) -> str:

        self.prompt_template = PromptTemplate.from_template(
            """
            Context : {context}
            As a specialized medical assistant for question-answering, your expertise lies in medicine. Each document is identified by its title, abstract, and retrieved chunks, cited with its ID in parentheses (e.g., (id)). Your goal is only to answer by yes or no. If unsure, simply acknowledge your inability to answer with maybe.
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
        for metadata in results["metadatas"][0]:
            if metadata["id"] not in id_list:
                id_list.append(metadata["id"])
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
        prompt_building_function=None,
    ) -> str:
        """Ask the RAG a question."""
        if prompt_building_function is None:
            prompt_building_function = self.build_prompt
        results = self.retrieve(question)
        prompt = prompt_building_function(question, results)
        print(prompt)
        return self.llm.ask(prompt)

    def ask_stream(
        self,
        question: str,
        web_search: bool = False,
        prompt_building_function=build_prompt,
    ) -> Generator:
        """Streams the response from the RAG."""
        results = self.retrieve(question)
        prompt = prompt_building_function(question, results)
        print(prompt)
        return self.llm.ask_stream(prompt, web_search=web_search)


# Example usage of HugChatLLM
if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_yaml(MAIN_DIR_PATH + "./config.yaml")

    # Initialize the Large Language Model (LLM)
    llm = HugChatLLM(config)

    # Initialize the SimpleRAG instance with the LLM and configuration
    rag = MetadataRAG(llm, config)

    # Load the collection named "test4_collection"
    rag.load_collection("test_collection")

    # Ingest a batch of documents into the collection
    # rag.ingest_batch(
    #     batch_size=16,
    #     doc_number=16,
    #     data_getter=get_data,
    #     doc_start=10500000,
    #     api=False,
    # )

    # Print the count of documents in the collection
    print(rag.collection.count())

    # Initialize variables to calculate the average length of documents
    total_length = 0
    num_documents = 0

    # Get the documents and metadata from the collection
    dict = rag.collection.get(include=["embeddings", "documents", "metadatas"])
    documents = dict["documents"]

    # Print an example document and its metadata
    print(documents[1])
    print(dict["metadatas"][1])

    # Calculate the total length of all documents and the number of documents
    for document in documents:
        total_length += len(document)
        num_documents += 1

    # Print the average length of documents in the collection
    print(total_length / num_documents)

    print(
        rag.ask(
            "What is the most common cause of death in the United States?",
        )
    )
