import evaluate

rouge = evaluate.load('rouge')


def get_rouge_score(predictions: list[str], targets: list[list[str]]) -> dict:
    res = rouge.compute(predictions=predictions, references=targets)
    return res

if __name__ == '__main__':
    preds = ["Transformers Transformers are fast plus efficient",
             "Good Morning", "I am waiting for new Transformers"]
    refs = [
        ["Transformers Transformers are fast plus efficient"],
        ["Good Morning Transformers", "Morning Transformers"],
        ["Good Morning", "I am waiting for new Transformers",
         "People are very excited about new Transformers"]
    ]
    result = rouge.compute(predictions=preds, references=refs)
    print(result)
