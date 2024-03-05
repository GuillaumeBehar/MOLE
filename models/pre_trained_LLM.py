import torch

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
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
        self.pipeline = pipeline("text-generation",
                                 model=self.model,
                                 tokenizer=self.tokenizer,
                                 torch_dtype=torch.bfloat16,
                                 device_map="auto",
                                 )

    def generate_w_pipeline(self, list_of_input_text: [str]):

        sequences = self.pipeline(
            list_of_input_text,
            max_new_tokens=10,
            return_full_text=False
        )
        return sequences

    def ask(self, sentence: str, web_search: bool = False) -> str:
        self.model.eval()
        inputs = self.tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            beam_output = self.model.generate(**inputs,
                                              min_length=100,
                                              max_length=256,
                                              num_beams=5,
                                              early_stopping=True
                                              )

        result = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
        return result


# def generate_from_biogpt(sentence: str, tokenizer, model) -> str:

#     inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
#     model.to("cuda")
#
#     with torch.no_grad():
#         beam_output = model.generate(**inputs,
#                                      min_length=100,
#                                      max_length=256,
#                                      num_beams=5,
#                                      early_stopping=True
#                                      )
#
#     # Move result back to CPU before decoding
#     result = tokenizer.decode(beam_output[0].to("cpu"), skip_special_tokens=True)
#     return result


if __name__ == "__main__":
    Bio = Biogpt(True, False, name='jpp')
    begin_prompt = "Answer with yes or no. Question: "
    text = "Are group 2 innate lymphoid cells ( ILC2s ) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?"
    end_prompt = " Answer: "
    output = Bio.generate_w_pipeline(begin_prompt+text+end_prompt)
    print(output)
    yesno_pipe = pipeline("text-classification")
    print(yesno_pipe(output[0]['generated_text']))
