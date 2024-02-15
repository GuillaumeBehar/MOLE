from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from models.hugchat_llm import *
from models.local_llm import *
from models.rag import *
from utils.utils import *


class SimpleRetrieveRAG(RAG):
    def __init__(self, llm, config) -> None:
        """Initialize the class with the provided large language model."""
        super().__init__(llm=llm, config=config, name="SimpleRetrieveRAG")
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64
        )
        self.prompt_template = PromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks. You are a medecin expert. Use the following retrieved context pieces to answer the question. If you don't know the answer, just say you don't know. Use up to three sentences and keep the response concise.
            Question : {question}
            Context : {context}
            Answer :
            """
        )

    def ingest(self, raw_documents: list[str]) -> None:
        """Ingest documents into the collection."""
        documents = self.text_splitter.create_documents(raw_documents)
        current_count = self.collection.count()
        self.collection.add(
            documents=[documents[k].page_content for k in range(len(documents))],
            ids=[str(k + current_count) for k in range(len(documents))],
        )

    def retrieve(self, question: str) -> str:
        """Retrieves relevant context for the given question."""
        results = self.collection.query(query_texts=[question], n_results=5)
        return results["documents"]

    def ask(self, question):
        documents = self.retrieve(question)
        context = "\n".join(documents[0])
        prompt = self.prompt_template.format(
            **{"context": context, "question": question}
        )
        print(prompt)
        return self.llm.ask(prompt)

    def ask_stream(self, question, web_search=False):
        documents = self.retrieve(question)
        context = "\n".join(documents[0])
        prompt = self.prompt_template.format(
            **{"context": context, "question": question}
        )
        print(prompt)
        return self.llm.ask_stream(prompt, web_search=web_search)


if __name__ == "__main__":
    config = load_yaml(MAIN_DIR_PATH + "./config.yaml")
    llm = HugChatLLM(config)
    rag = SimpleRetrieveRAG(llm, config)
    rag.load_collection("test_collection")
    # rag.ingest_folder(MAIN_DIR_PATH + rag.config["data_directory"])

    total_length = 0
    num_documents = 0
    dict = rag.collection.get(include=["embeddings", "documents", "metadatas"])
    documents = dict["documents"]

    for document in documents:
        total_length += len(document)
        num_documents += 1

    print(total_length / num_documents)
