import requests
import pandas as pd
import json
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed

from models.pre_trained_LLM import generate_from_biogpt
from models.pre_trained_LLM import Biogpt
from rouge import get_rouge_score


import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
from models.llm import LLM


API_URL = "https://huggingface.co/api/datasets/pubmed_qa/parquet/pqa_artificial/train"


def query(api):
    response = requests.get(api)
    return response.json()


url_parquet = query(API_URL)[0]
df = pd.read_parquet(url_parquet)


def get_pmid_list(json_name: str) -> None:
    pubid_list = df['pubid'].tolist()
    json_data = {'pubid_list': pubid_list}
    with open(json_name, 'w') as json_file:
        json.dump(json_data, json_file)


def get_instance(i: int) -> None:
    row = df.head(i)
    return row.iloc[0]


def evaluate_long(llm: LLM, n_instances: int, show: bool) -> str | dict:
    generated = []
    targets = []
    for i in range(min(n_instances, len(df))):
        instance = df.iloc[i]
        question = instance['question']
        long_answer = [instance['long_answer']]
        output = llm.ask(question)
        generated.append(output)
        targets.append(long_answer)
        if show:
            print(f'\nInstance {i + 1}:')
            print(f'Generated Answer: {output}')
            print(f'Expected Answer: {long_answer}')
            print(get_rouge_score([output], [long_answer]))

    res = get_rouge_score(predictions=generated, targets=targets)
    return res


if __name__ == "__main__":
    # print(df.head(0))
    biogpt = Biogpt(True, False, name='jpp')
    print(evaluate_long(llm=biogpt, n_instances=3, show=True))
