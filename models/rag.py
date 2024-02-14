import chromadb
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from models.hugchat_llm import *
from models.local_llm import *


class RAG:
    def __init__(self, llm, config, name) -> None:
        """Initialize the class with the provided large language model."""
        self.name = name
        self.llm = llm
        self.config = config
        self.embedding_function = None
        self.text_splitter = None
        self.prompt_template = None
        self.chroma_client = chromadb.PersistentClient(config["collections_directory"])

    def load_collections(self, collection_name):
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except Exception as e:
            print("Error loading collection: %s", e)
            self.collection = self.chroma_client.create_collection(name=collection_name)

    def ask(self, question: str) -> str:
        """Ask the RAG a question."""
        pass  # This method will be implemented in subclasses


class SimpleRetrieveRAG(RAG):
    def __init__(self, llm, config) -> None:
        """Initialize the class with the provided large language model."""
        super().__init__(llm=llm, config=config, name="SimpleRetrieveRAG")
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64
        )
        self.prompt_template = PromptTemplate.from_template(
            """
            <s> [INST] Vous êtes un assistant pour les tâches de question-réponse. Utilisez les morceaux de contexte récupérés suivants pour répondre à la question. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. Utilisez au maximum trois phrases et gardez la réponse concise. [/INST] </s>
            [INST] Question : {question}
            Contexte : {context}
            Réponse : [/INST]
        """
        )
        self.load_collections("rag")

    def retrieve(self, question: str) -> str:
        """
        Retrieves relevant context for the given question.
        """
        results = self.collection.query(query_texts=[question], n_results=5)
        return results["documents"]

    def ask(self, question):
        prompt = self.prompt_template.format(
            **{"context": self.retrieve(question), "question": question}
        )
        return self.llm.ask(prompt)

    def ask_stream(self, question, web_search=False):
        prompt = self.prompt_template.format(
            **{"context": self.retrieve(question), "question": question}
        )
        return self.llm.ask_stream(prompt, web_search=web_search)


if __name__ == "__main__":
    config = load_yaml(MAIN_DIR_PATH + "./config.yaml")
    llm = HugChatLLM(config)
    rag = RAG(llm)
    question = "Quel est le nom de la Reine ?"
    print(rag.ask_stream(question))


class ChatBot:

    def __init__(self, llm) -> None:
        """Initialize the class with the provided local language model."""
        self.llm = llm
        self.vector_store = None
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.retriever = None
        self.chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64
        )
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] Vous êtes un assistant pour les tâches de question-réponse. Utilisez les morceaux de contexte récupérés suivants pour répondre à la question. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. Utilisez au maximum trois phrases et gardez la réponse concise. [/INST] </s>
            [INST] Question : {question}
            Contexte : {context}
            Réponse : [/INST]
            """
        )

    def ingest(self, pdf_file_path: str) -> None:
        """Ingests a PDF file and processes its contents."""

        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks, embedding=self.embedding_function
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.3,
            },
        )
        if self.model.local:
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm.model.ask
                | StrOutputParser()
            )
        else:
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm.model.ask_stream
                | StrOutputParser()
            )

    def ask(self, question: str) -> None:
        if not self.chain:
            if self.model.local:
                return self.llm.model.ask(question)
            else:
                return self.llm.model.ask_stream(question)
        return self.chain.invoke(question)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
