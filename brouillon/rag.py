from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.schema.output_parser import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from local_llm import *


class ChatBot:

    def __init__(self, local_LLM: LocalLLM) -> None:
        """Initialize the class with the provided local language model."""
        self.model = local_LLM.model
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
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def ask(self, query: str) -> None:
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
