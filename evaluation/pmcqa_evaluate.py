import requests
import pandas as pd
import json
import random

from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt


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

# from utils.custom_utils import load_yaml
# from models.hugchat_llm import HugChatLLM
# from models.llm import LLM
from models.pre_trained_LLM import *
from evaluation.our_metrics import get_all_scores, get_rouge1_score, get_cm

API_URL = "https://huggingface.co/api/datasets/pubmed_qa/parquet/pqa_artificial/train"


def query(api):
    response = requests.get(api)
    return response.json()


url_parquet = query(API_URL)[0]
EVALUATION_DATAFRAME = pd.read_parquet(url_parquet)
EVALUATION_DATAFRAME.set_index('pubid', inplace=True)


def get_pmid_list(json_name: str, n_instance: int) -> list[int]:
    pubid_list = EVALUATION_DATAFRAME.index.tolist()
    n_pubid_list = random.sample(pubid_list, n_instance)
    json_data = {"pubid_list": n_pubid_list}
    with open(json_name, "w") as json_file:
        json.dump(json_data, json_file)
    return n_pubid_list


def get_instance_from_pubid(pubid: int) -> 'pd.core.series.Series':
    row = EVALUATION_DATAFRAME.loc[pubid]
    return row

def answers_generation(llm: LLM, id_instances_list: list, json_filename: str) -> dict:
    generated_dict = {}

    for id in tqdm(id_instances_list):
        instance = get_instance_from_pubid(id)
        question = instance["question"]
        long_answer = instance["long_answer"]
        generated_output = llm.ask(question)
        dict_instance = {"generated_answer": generated_output,
                         "target_answer": long_answer,
                         "final_decision_target": instance["final_decision"]
                         }
        generated_dict[id] = dict_instance

    with open(json_filename, 'w') as json_file:
        json.dump(generated_dict, json_file, indent=4)
    return generated_dict


# def yesno_from_answer(llm: LLM, question: str, answer: str) -> str:
#
#     return final_answer


def evaluate_long(llm: LLM, id_instances_list: list, show: bool) -> str | dict:
    generated = []
    targets = []

    for id in tqdm(id_instances_list):
        instance = get_instance_from_pubid(id)
        question = instance["question"]
        long_answer = [instance["long_answer"]]
        generated_output = llm.ask(question)
        generated.append(generated_output)
        targets.append(long_answer)
        if show:
            print(f"Generated Answer: {generated_output}")
            print(f"Expected Answer: {long_answer[0]}")
            print(get_all_scores([generated_output], [long_answer]))

    results = get_all_scores(predictions=generated, targets=targets)
    return results


def get_yesno(answer: str) -> str:
    if answer == " Yes.":
        return "yes"
    elif answer == " No.":
        return "no"
    else:
        print(answer)
        return answer


def evaluate_short(llm: LLM, id_instances_list: list, show: bool) -> dict:
    generated = []
    targets = []
    for id in tqdm(id_instances_list):
        instance = get_instance_from_pubid(id)
        question = instance["question"]
        yesno_answer = instance["final_decision"]
        generated_answer = get_yesno(llm.ask(question))
        generated.append(generated_answer)
        targets.append(yesno_answer)
        if show and yesno_answer[0] == 'no':
            print(f"\nInstance {question}:")
            print(f"Generated Answer: {generated_answer}")
            print(f"Expected Answer: {yesno_answer[0]}")
            print(f'rouge: {get_rouge1_score([generated_answer], [yesno_answer])}')
    results = get_cm(generated, targets)
    return results


if __name__ == "__main__":
    list_of_id = get_pmid_list('list.json', n_instance=50)
    biogpt = Biogpt(True, False, name="jpp")
    answers_generated = answers_generation(biogpt, list_of_id, 'biogpt_answers.json')
    # scores = evaluate_short(
    #     llm=biogpt,
    #     id_instances_list=list_of_id,
    #     show=True)
    # print(f'Scores: {scores}')
    # cm = scores.get("confusion_matrix")
    # cm_display = ConfusionMatrixDisplay(cm).plot()
    # plt.show()

