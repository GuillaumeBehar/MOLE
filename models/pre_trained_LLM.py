import torch
import spacy
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from llm import LLM


class Biogpt(LLM):
    def __init__(self, local: bool, loaded: bool, name: str):
        super().__init__(local=True, loaded=False, name=None)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt").to(torch.device("cuda"))
        self.pipeline = pipeline("text-generation",
                                 model=self.model,
                                 tokenizer=self.tokenizer,
                                 torch_dtype=torch.bfloat16,
                                 device_map="cuda",
                                 )

    def ask(self, input_text: str):
        begin_prompt = "Question: "
        end_prompt = ("The answer of the question is")

        sequences = self.pipeline(
            begin_prompt + input_text + end_prompt,
            max_new_tokens=100,
            return_full_text=False
        )
        return sequences[0]['generated_text']

    # def ask(self, sentence: str, web_search: bool = False) -> str:
    #     self.model.eval()
    #     inputs = self.tokenizer(sentence, return_tensors="pt")
    #
    #     with torch.no_grad():
    #         beam_output = self.model.generate(**inputs,
    #                                           min_length=100,
    #                                           max_length=256,
    #                                           num_beams=5,
    #                                           early_stopping=True
    #                                           )
    #
    #     result = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
    #     return result


def generate_yesno_from_biogpt(generated_dict: dict, value_to_evaluate: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt").to(torch.device("cuda"))
    # pipe = pipeline("question-answering",
    #                     model=model,
    #                     tokenizer=tokenizer,
    #                     torch_dtype=torch.bfloat16,
    #                     device_map="cuda",
    #                     )
    pipe = pipeline("question-answering",
                    model="dmis-lab/biobert-large-cased-v1.1-squad",
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                    )

    for key, value in generated_dict.items():
        question = value["question"]
        long_answer = value[value_to_evaluate]
        short_answer = pipe(
            question=question,
            context=long_answer,
            max_new_tokens=100,
            return_full_text=False
        )['answer']
        value["short_generated"] = short_answer
    return generated_dict

# Charger le modèle spaCy pour l'anglais
nlp = spacy.load("en_core_web_sm")


def analyse_neg(texte):
    doc = nlp(texte)

    # Exemple d'analyse : vérifier si le texte est à la forme négative
    est_negatif = any(token.dep_ == "neg" for token in doc)

    return est_negatif


if __name__ == "__main__":
    Bio = Biogpt(True, False, name='jpp')
    text = ("Does vagus nerve contribute to the development of steatohepatitis and obesity in phosphatidylethanolamine "
            "N-methyltransferase deficient mice?")
    output = Bio.ask(text)
    print(output)
    yesno_pipe = pipeline("text-classification")
    print(yesno_pipe(output))
    print(analyse_neg(output))