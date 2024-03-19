from langchain.text_splitter import RecursiveCharacterTextSplitter
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
from models.groq_llm import GroqLLM
from evaluation.questions_test import id_test_pmc, id_test_pm
from models.metadata_rag import MetadataRAG
from models.simple_rag import SimpleRAG
from utils.lecture_xml import get_data
from models.llm import LLM


def create_collections():

    llm = LLM(True, True, "llm")
    rag = MetadataRAG(llm, config)

    doc_count = 1000
    chunk_size = 768
    batch_size = 12
    try:
        rag.chroma_client.delete_collection(f"only_QA")
    except:
        pass
    rag.load_collection(f"only_QA")
    rag.ingest_list(
        batch_size=batch_size,
        id_list=id_test_pmc,
        data_getter=get_data,
        api=True,
        show=True,
    )
    print("Number of chunk in the collection:", rag.collection.count())

    doc_count = 1000
    chunk_size = 256
    batch_size = 4
    rag.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=64)
    try:
        rag.chroma_client.delete_collection(f"{chunk_size}_{doc_count}")
    except:
        pass
    rag.load_collection(f"{chunk_size}_{doc_count}")
    rag.ingest_list(
        batch_size=batch_size,
        id_list=id_test_pmc,
        data_getter=get_data,
        api=True,
        show=True,
    )
    rag.ingest_batch(
        batch_size=batch_size,
        doc_number=doc_count-50,
        data_getter=get_data,
        doc_start=10500000,
        api=False,
        show=True,
    )
    print("Number of chunk in the collection:", rag.collection.count())

    doc_count = 950
    chunk_size = 256
    batch_size = 4
    rag.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=64)
    try:
        rag.chroma_client.delete_collection(f"{chunk_size}_{doc_count}")
    except:
        pass
    rag.load_collection(f"{chunk_size}_{doc_count}")
    rag.ingest_batch(
        batch_size=batch_size,
        doc_number=doc_count,
        data_getter=get_data,
        doc_start=10500000,
        api=False,
        show=True,
    )
    print("Number of chunk in the collection:", rag.collection.count())

    doc_count = 1000
    chunk_size = 512
    batch_size = 8
    rag.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=64)
    try:
        rag.chroma_client.delete_collection(f"{chunk_size}_{doc_count}")
    except:
        pass
    rag.load_collection(f"{chunk_size}_{doc_count}")
    rag.ingest_list(
        batch_size=batch_size,
        id_list=id_test_pmc,
        data_getter=get_data,
        api=True,
        show=True,
    )
    rag.ingest_batch(
        batch_size=batch_size,
        doc_number=doc_count-50,
        data_getter=get_data,
        doc_start=10500000,
        api=False,
        show=True,
    )
    print("Number of chunk in the collection:", rag.collection.count())

    doc_count = 950
    chunk_size = 512
    batch_size = 8
    rag.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=64)
    try:
        rag.chroma_client.delete_collection(f"{chunk_size}_{doc_count}")
    except:
        pass
    rag.load_collection(f"{chunk_size}_{doc_count}")
    rag.ingest_batch(
        batch_size=batch_size,
        doc_number=doc_count,
        data_getter=get_data,
        doc_start=10500000,
        api=False,
        show=True,
    )
    print("Number of chunk in the collection:", rag.collection.count())

    doc_count = 1000
    chunk_size = 768
    batch_size = 12
    rag.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=64)
    try:
        rag.chroma_client.delete_collection(f"{chunk_size}_{doc_count}")
    except:
        pass
    rag.load_collection(f"{chunk_size}_{doc_count}")
    rag.ingest_list(
        batch_size=batch_size,
        id_list=id_test_pmc,
        data_getter=get_data,
        api=True,
        show=True,
    )
    rag.ingest_batch(
        batch_size=batch_size,
        doc_number=doc_count-50,
        data_getter=get_data,
        doc_start=10500000,
        api=False,
        show=True,
    )
    print("Number of chunk in the collection:", rag.collection.count())

    doc_count = 950
    chunk_size = 768
    batch_size = 12
    rag.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=64)
    try:
        rag.chroma_client.delete_collection(f"{chunk_size}_{doc_count}")
    except:
        pass
    rag.load_collection(f"{chunk_size}_{doc_count}")
    rag.ingest_batch(
        batch_size=batch_size,
        doc_number=doc_count,
        data_getter=get_data,
        doc_start=10500000,
        api=False,
        show=True,
    )
    print("Number of chunk in the collection:", rag.collection.count())


if __name__ == "__main__":

    config = load_yaml(MAIN_DIR_PATH + "./config.yaml")

    # create_collections()

    # llm = LocalGgufLLM(config)
    # llm = HugChatLLM(config)
    llm = GroqLLM(config)
    model_id = 0
    rag = SimpleRAG(llm, config)
    rag.name += "_gemma"

    for doc_count in [950, 1000]:
        for chunk_size in [256, 512, 768]:
            rag.load_collection(f"{chunk_size}_{doc_count}")
            print(
                f"Number of chunk in the collection {chunk_size}_{doc_count}:", rag.collection.count())

    llm.load_model(model_id)
    for doc_count in [950, 1000]:
        for chunk_size in [512]:
            rag.load_collection(f"{chunk_size}_{doc_count}")
            print("Number of chunk in the collection:", rag.collection.count())
            scores = evaluate_long(
                llm=rag, id_instances_list=id_test_pm, show=False)
            print(f'Average Scores: {scores}')
            with open(f"{rag.name}_{chunk_size}_{doc_count}.json", "w") as f:
                json.dump(scores, f)

    rag = MetadataRAG(llm, config)
    rag.name += "_gemma"

    for doc_count in [950, 1000]:
        for chunk_size in [512]:
            rag.load_collection(f"{chunk_size}_{doc_count}")
            print("Number of chunk in the collection:", rag.collection.count())
            scores = evaluate_long(
                llm=rag, id_instances_list=id_test_pm, show=False)
            print(f'Average Scores: {scores}')
            with open(f"{rag.name}_{chunk_size}_{doc_count}.json", "w") as f:
                json.dump(scores, f)
