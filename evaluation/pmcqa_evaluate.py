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
    filtered_df = EVALUATION_DATAFRAME.loc[EVALUATION_DATAFRAME['final_decision'] == 'no']
    pubid_list = filtered_df.index.tolist()
    #n_pubid_list = random.sample(pubid_list, n_instance)
    json_data = {"pubid_list": pubid_list}
    with open(json_name, "w") as json_file:
        json.dump(json_data, json_file)
    return pubid_list


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
        dict_instance = {"question": question,
                         "generated_answer": generated_output,
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
    if answer in ['yes', 'Yes', 'Yes.', 'yes,','yes.','Yes,']:
        return "yes"
    elif answer in ['no', 'no\n','No', 'no.', 'No.', 'No,', 'no,']:
        return "no"
    else:
        return "other"


def evaluate_short_from_dict(generated_dict: dict) -> dict:
    generated = []
    targets = []
    for instance in generated_dict.values():
        yesno_generated = get_yesno(instance["short_generated"])
        generated.append(yesno_generated)
        targets.append(instance["final_decision_target"])
    noisy_answers = generated.count("other")/len(generated)
    results = get_cm(generated, targets)
    results["noisy_answers_rate"] = noisy_answers*100
    return results

def get_dict_for_yesno_evaluation(df: pd.DataFrame, n: int) -> dict:
    df_subset = df[['question', 'long_answer', 'final_decision']].sample(n=n)
    yesno_evaluation_dict = df_subset.to_dict(orient='index')
    with open('DataForYesNoEvaluation1000.json','w') as f:
        json.dump(yesno_evaluation_dict, f, indent=4)
    return yesno_evaluation_dict

if __name__ == "__main__":
    # list_of_id = get_pmid_list('list.json', n_instance=500)

    # list_of_id = [22022005, 21639875, 20852029, 25844699, 25379003, 26817669, 22563393, 20659337, 27643685, 26693009,
    #               19888227, 20878146, 26337974, 23355459, 25495800, 22640485, 24059973, 24409166, 22909062, 24447369,
    #               19180231, 22569336, 23231769, 23557178,
    #               21617180, 24958351, 27500275, 19933996, 24330812, 26227965, 27574676, 27473420, 22709483, 26289293,
    #               23949151, 27336604, 26460750, 18575589, 24884655, 18493326, 23015864, 26175775, 26418562, 26418133,
    #               21696606, 25036418, 24847033, 26295946, 27595989, 21981946]
    # print(len(list_of_id))
    # biogpt = Biogpt(True, False, name="jpp")
    # answers_generated = answers_generation(biogpt, list_of_id, 'biogpt_answers2.json')

    with open('biogpt_50decisions.json', 'r') as json_file:
        data = json.load(json_file)

    Mixtral_evaluation = evaluate_short_from_dict(data)
    print(Mixtral_evaluation)
    cm = Mixtral_evaluation.get("confusion_matrix")
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.show()

    # results_dict = generate_yesno_from_biogpt(data, "target_answer")
    # with open('final_decisions.json', 'w') as json_file:
    #     json.dump(results_dict, json_file, indent=4)

    # scores = evaluate_short(
    #     llm=biogpt,
    #     id_instances_list=list_of_id,
    #     show=True)
    # print(f'Scores: {scores}')
    # cm = scores.get("confusion_matrix")
    # cm_display = ConfusionMatrixDisplay(cm).plot()
    # plt.show()

    id_test_pm = [22022005, 21639875, 20852029, 25844699, 25379003, 26817669, 22563393, 20659337, 27643685, 26693009,
                  19888227, 20878146, 26337974, 23355459, 25495800, 22640485, 24059973, 24409166, 22909062, 24447369,
                  19180231, 22569336, 23231769, 23557178,
                  21617180, 24958351, 27500275, 19933996, 24330812, 26227965, 27574676, 27473420, 22709483, 26289293,
                  23949151, 27336604, 26460750, 18575589, 24884655, 18493326, 23015864, 26175775, 26418562, 26418133,
                  21696606, 25036418, 24847033, 26295946, 27595989, 21981946]







