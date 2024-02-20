from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from models.rag import *


class SimpleRAG(RAG):
    def __init__(self, llm: LLM, config: dict) -> None:
        """Initialize the class with the provided large language model."""
        super().__init__(llm=llm, config=config, name="SimpleRetrieveRAG")
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2", device="cuda"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=64
        )
        self.prompt_template = PromptTemplate.from_template(
            """
            As a specialized medical assistant for question-answering, your expertise lies in the field of medicine. Utilizing the provided context pieces, you're tasked with delivering accurate responses. If uncertain, simply acknowledge that you're unable to provide an answer. Responses should be succinct, comprising up to three sentences for clarity and efficiency.
            Question : {question}
            Context : {context}
            Answer :
            """
        )

    def ingest(self, raw_documents: list[str]) -> None:
        """Ingest documents into the collection."""
        documents = self.text_splitter.create_documents(
            [document["text"] for document in raw_documents]
        )
        current_count = self.collection.count()
        self.collection.add(
            documents=[document.page_content for document in documents],
            ids=[str(k + current_count) for k in range(len(documents))],
        )

    def retrieve(self, question: str) -> str:
        """Retrieves relevant context for the given question."""
        results = self.collection.query(query_texts=[question], n_results=5)
        return results["documents"]

    def ask(self, question: str) -> str:
        documents = self.retrieve(question)
        context = "\n".join(documents[0])
        prompt = self.prompt_template.format(
            **{"context": context, "question": question}
        )
        print(prompt)
        return self.llm.ask(prompt)

    def ask_stream(self, question, web_search=False) -> Generator:
        documents = self.retrieve(question)
        context = "\n".join(documents[0])
        prompt = self.prompt_template.format(
            **{"context": context, "question": question}
        )
        print(prompt)
        return self.llm.ask_stream(prompt, web_search=web_search)


if __name__ == "__main__":
    config = load_yaml(MAIN_DIR_PATH + "./config.yaml")
    llm = LLM(False, True, "test")
    rag = SimpleRAG(llm, config)
    rag.load_collection("test4_collection")

    rag.ingest_batch(
        batch_size=16,
        doc_number=176288,
        data_getter=get_data,
        doc_start=10500000,
        api=False,
    )

    print(rag.collection.count())

    total_length = 0
    num_documents = 0
    dict = rag.collection.get(include=["embeddings", "documents", "metadatas"])
    documents = dict["documents"]
    print(documents[1])
    print(dict["metadatas"][1])

    for document in documents:
        total_length += len(document)
        num_documents += 1

    print(total_length / num_documents)
