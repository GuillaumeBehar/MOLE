from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from typing import Generator


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

    def ingest(self, raw_documents: list[dict]) -> None:
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

    def ask(self, question: str, web_search: bool = False) -> str:
        """Ask the RAG a question."""
        documents = self.retrieve(question)
        context = "\n".join(documents[0])
        prompt = self.prompt_template.format(
            **{"context": context, "question": question}
        )
        print("Length of prompt:", len(prompt))
        print(prompt)
        return self.llm.ask(prompt)

    def ask_stream(self, question: str, web_search: bool = False) -> Generator:
        """Streams the response from the RAG."""
        documents = self.retrieve(question)
        context = "\n".join(documents[0])
        prompt = self.prompt_template.format(
            **{"context": context, "question": question}
        )
        print("Length of prompt:", len(prompt))
        print(prompt)

        return self.llm.ask_stream(prompt, web_search=web_search)


# Example usage of HugChatLLM
if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_yaml(MAIN_DIR_PATH + "./config.yaml")

    # Initialize the Large Language Model (LLM)
    llm = LLM(False, True, "test")

    # Initialize the SimpleRAG instance with the LLM and configuration
    rag = SimpleRAG(llm, config)

    # Load the collection named "test4_collection"
    rag.load_collection("test_collection")

    # Ingest a batch of documents into the collection
    rag.ingest_batch(
        batch_size=4,
        doc_start=10500000,
        doc_number=1000,
        data_getter=get_data,
        api=False,
        show=True,
    )

    # Ingest a list of documents into the collection
    # rag.ingest_list(
    #     batch_size=32,
    #     id_list=EVALUATION_DATAFRAME["pubid"].tolist()[:1000],
    #     data_getter=get_data,
    #     api=False,
    #     show=True,
    # )

    # Print the count of documents in the collection
    print("Number of chunk in the collection", rag.collection.count())
