import requests
import pandas as pd
import json
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed

from models.pre_trained_LLM import generate_from_biogpt
from rouge import get_rouge_score

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


def evaluate_long(model_name: str, show: bool) -> str | dict:
    instance = df.head(1)
    question = instance['question'].values[0]
    long_answer = instance['long_answer'].values[0]
    if model_name == 'biogpt':
        Tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        Model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        output = generate_from_biogpt(question, Tokenizer, Model)
        if show:
            print(f'answer generated: {output}\n')
            print(f'answer expected; {long_answer}')
    else:
        return 'Model not found'
    result = get_rouge_score(predictions=[output], targets=[long_answer])
    return result


if __name__ == "__main__":
    print(evaluate_long(model_name='biogpt', show=True))
