import evaluate

rouge = evaluate.load('rouge')


def get_rouge_score(predictions: list[str], targets: list[str]) -> dict:
    results = rouge.compute(predictions=predictions, references=targets)
    return results


if __name__ == '__main__':
    preds = ["Transformers Transformers are fast plus efficient",
             "Good Morning", "I am waiting for new Transformers"]
    refs = [
        ["HuggingFace Transformers are fast efficient plus awesome",
         "Transformers are awesome because they are fast to execute"],
        ["Good Morning Transformers", "Morning Transformers"],
        ["People are eagerly waiting for new Transformer models",
         "People are very excited about new Transformers"]
    ]
    results = rouge.compute(predictions=preds, references=refs)
    print(type(results))
