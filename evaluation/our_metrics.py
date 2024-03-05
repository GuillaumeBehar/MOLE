from evaluate import load
import numpy as np

bleu = load("bleu")
rouge = load("rouge")
bert = load("bertscore")


def get_rouge_score(predictions: list[str], targets: list[list[str]]) -> dict:
    result = rouge.compute(predictions=predictions,
                           references=targets,
                           rouge_types=['rouge1', 'rouge2', 'rougeL'])
    return result


def get_bertscore(predictions: list[str], targets: list[list[str]]) -> dict:
    result = bert.compute(predictions=predictions,
                          references=targets,
                          model_type="bert-base-uncased")
    res = {"avg_precision": np.mean(result["precision"]),
           "avg_recall": np.mean(result["recall"]),
           "avg_f1": np.mean(result["f1"])
           }
    return res


def get_bleu_score(predictions: list[str], targets: list[list[str]]) -> dict:
    result = bleu.compute(predictions=predictions, references=targets)
    res = {"bleu": result["bleu"],
           "avg_precision": np.mean(result["precisions"])
           }
    return res


def get_all_scores(predictions: list[str], targets: list[list[str]]) -> dict:
    print('compute bleu score')
    bleu_res = get_bleu_score(predictions, targets)

    print('compute rouge score')
    rouge_res = get_rouge_score(predictions, targets)

    print('compute bertscore')
    bert_res = get_bertscore(predictions, targets)
    res = {'rouge': rouge_res,
           'bleu': bleu_res,
           'bertscore': bert_res
           }
    return res


if __name__ == '__main__':
    preds = ["Transformers Transformers are fast plus efficient",
             "Good Morning", "I am waiting for new Transformers"]
    refs = [
        ["Transformers Transformers are fast plus efficient"],
        ["I love Transformers", "They hate chocolate"],
        ["Good Morning", "I am waiting for new Transformers",
         "People are very excited about new Transformers"]
    ]
    scores = get_all_scores(predictions=preds, targets=refs)
    print(scores)
