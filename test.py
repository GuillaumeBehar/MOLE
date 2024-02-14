import chromadb
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
from utils import *
from hugchat_llm import HugChatLLM

MAIN_DIR_PATH = up(up(os.path.abspath(__file__)))


# raw_documents = TextLoader(MAIN_DIR_PATH + "./La_Reine_Margot.txt").load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)

# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # db = Chroma.from_documents(
# #     documents,
# #     embedding_function,
# #     persist_directory=MAIN_DIR_PATH + "./databases/",
# # )

# db = Chroma.from_persisted(MAIN_DIR_PATH + "./databases/")

# query = "Quel est le nom de la Reine ?"

# docs = db.similarity_search(query)

# print(len(docs))


chroma_client = chromadb.PersistentClient(MAIN_DIR_PATH + "./databases/")
# collection = chroma_client.create_collection(name="my_collection")
# raw_documents = TextLoader(MAIN_DIR_PATH + "./La_Reine_Margot.txt").load()
# text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
# collection.add(
#     documents=[documents[k].page_content for k in range(len(documents))],
#     ids=[str(k) for k in range(len(documents))],
# )

collection = chroma_client.get_collection(name="my_collection")


# data = {
#     "documents": list[str]
#     "metadata": list[dict],
#     "ids": list[str],
# }

results = collection.query(query_texts=["Quel est le nom de la Reine ?"], n_results=5)
print(results["documents"])

config = load_yaml(MAIN_DIR_PATH + "./config.yaml")
hugchat_llm = HugChatLLM(config)

question = "Quel est le nom de la Reine ?"
prompt = f"""
Vous êtes un assistant pour les tâches de question-réponse. Vous devez utiliser les morceaux de contexte spéciaux suivants pour repondre à la question. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. Restez convaincus. Utilisez au maximum trois phrases et gardez la réponse concise. Répondez en français.
Question : {question}
Context : {results["documents"]}
Réponse en français :
"""

print(prompt)

response = hugchat_llm.ask(prompt)
print("Response:", response)
