from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
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

    def ingest(self, raw_documents) -> None:
        """Ingest documents into the collection."""
        documents = self.text_splitter.split_documents(raw_documents)
        self.collection.add(
            documents=[documents[k].page_content for k in range(len(documents))],
            ids=[str(k) for k in range(len(documents))],
        )

    def retrieve(self, question: str) -> str:
        """Retrieves relevant context for the given question."""
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
    rag = SimpleRetrieveRAG(llm, config)
    rag.load_collection("rag")
    question = "Quel est le nom de la Reine ?"
    for word in rag.ask_stream(question):
        print(word)
