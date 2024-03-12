from langchain.text_splitter import RecursiveCharacterTextSplitter
from time import perf_counter
import json


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

from utils.custom_utils import load_yaml
from evaluation.pmcqa_evaluate import evaluate_long
from models.hugchat_llm import HugChatLLM
from models.local_gguf_llm import LocalGgufLLM
from evaluation.questions_test import id_test_pmc, id_test_pm
from models.metadata_rag import MetadataRAG
from models.simple_rag import SimpleRAG
from utils.lecture_xml import get_data

if __name__ == "__main__":
    config = load_yaml(MAIN_DIR_PATH + "./config.yaml")

    # llm = LocalGgufLLM(config)
    llm = HugChatLLM(config)
    rag = SimpleRAG(llm, config)

    # Chunk size 256, number of documents 1000
    chunck_size = 256
    rag.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunck_size, chunk_overlap=64)
    rag.load_collection(f"{chunck_size}")
    start_time = perf_counter()
    # rag.ingest_list(
    #     batch_size=8,
    #     id_list=id_test_pmc,
    #     data_getter=get_data,
    #     api=False,
    #     show=False,
    # )
    # rag.ingest_batch(
    #     batch_size=8,
    #     doc_number=950,
    #     data_getter=get_data,
    #     doc_start=10500000,
    #     api=False,
    #     show=False,
    # )
    end_time = perf_counter()
    print("Number of chunk in the collection:", rag.collection.count())
    print("Time to ingest the collection:", end_time - start_time, "s")
    llm.load_model(5)
    scores = evaluate_long(
        llm=rag, id_instances_list=id_test_pm, show=False)
    print(f'Average Scores: {scores}')
    llm.unload_model()
    with open(f"scores_{chunck_size}_collection_1000.json", "w") as f:
        json.dump(scores, f)

    # Chunk size 256, number of documents 950
    chunck_size = 256
    rag.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunck_size, chunk_overlap=64)
    rag.load_collection(f"{chunck_size}_collection_950")
    start_time = perf_counter()
    # rag.ingest_batch(
    #     batch_size=8,
    #     doc_number=950,
    #     data_getter=get_data,
    #     doc_start=10500000,
    #     api=False,
    #     show=False,
    # )
    end_time = perf_counter()
    print("Number of chunk in the collection:", rag.collection.count())
    print("Time to ingest the collection:", end_time - start_time, "s")
    llm.load_model(5)
    scores = evaluate_long(
        llm=rag, id_instances_list=id_test_pm, show=False)
    print(f'Average Scores: {scores}')
    llm.unload_model()
    with open(f"scores_{chunck_size}_collection_950.json", "w") as f:
        json.dump(scores, f)
