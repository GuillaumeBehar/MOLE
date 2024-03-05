import requests
import pandas as pd
import json
from datasets import load_metric


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
from models.hugchat_llm import HugChatLLM
from models.llm import LLM
from models.pre_trained_LLM import *
from evaluation.rouge import get_rouge_score


API_URL = "https://huggingface.co/api/datasets/pubmed_qa/parquet/pqa_artificial/train"


def query(api):
    response = requests.get(api)
    return response.json()


url_parquet = query(API_URL)[0]
EVALUATION_DATAFRAME = pd.read_parquet(url_parquet)


def get_pmid_list(json_name: str) -> None:
    pubid_list = EVALUATION_DATAFRAME["pubid"].tolist()
    json_data = {"pubid_list": pubid_list}
    with open(json_name, "w") as json_file:
        json.dump(json_data, json_file)


def get_instance(i: int) -> None:
    row = EVALUATION_DATAFRAME.head(i)
    return row.iloc[0]


def evaluate_long(llm: LLM, n_instances: int, show: bool) -> str | dict:
    metric = load_metric('glue', 'mrpc')
    generated = []
    targets = []
    for i in range(min(n_instances, len(EVALUATION_DATAFRAME))):
        instance = EVALUATION_DATAFRAME.iloc[i]
        question = instance["question"]
        long_answer = [instance["long_answer"]]
        output = llm.ask(question)
        generated.append(output)
        targets.append(long_answer)
        if show:
            print(f"\nInstance {i + 1}:")
            print(f"Generated Answer: {output}")
            print(f"Expected Answer: {long_answer}")
            print(get_rouge_score([output], [long_answer]))

    res = get_rouge_score(predictions=generated, targets=targets)
    return res


def evaluate_short(llm: LLM, n_instances: int, show: bool) -> str | dict:
    begin_prompt = "Answer by yes or no. Question: "
    end_prompt = " Answer : "
    for i in range(min(n_instances, len(EVALUATION_DATAFRAME))):
        instance = EVALUATION_DATAFRAME.iloc[i]
        question = instance["question"]
        long_answer = [instance["final_decision"]]
        output = llm.ask(begin_prompt+question+end_prompt)
        if show:
            print(f"\nInstance {i + 1}:")
            print(f"Generated Answer: {output}")
            print(f"Expected Answer: {long_answer}")
            print(get_rouge_score([output], [long_answer]))


if __name__ == "__main__":

    biogpt = Biogpt(True, False, name="jpp")
    # config = load_yaml(MAIN_DIR_PATH + "./config.yaml")
    #
    # hugchat_llm = HugChatLLM(config)
    evaluate_long(llm=biogpt, n_instances=3, show=True)
