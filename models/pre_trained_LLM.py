import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from llm import LLM


class Biogpt(LLM):
    def __init__(self, local: bool, loaded: bool, name: str):
        super().__init__(local=True, loaded=False, name=None)
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

    def ask(self, sentence: str, web_search: bool = False) -> str:
        inputs = self.tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            beam_output = self.model.generate(
                **inputs,
                min_length=100,
                max_length=256,
                num_beams=5,
                early_stopping=True
            )

        # Move result back to CPU before decoding
        result = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
        return result


def generate_from_biogpt(
    sentence: str, tokenizer: BioGptTokenizer, model: BioGptForCausalLM
) -> str:
    # Move inputs and model to GPU
    inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
    model.to("cuda")

    with torch.no_grad():
        # Use CUDA for generation
        beam_output = model.generate(
            **inputs, min_length=100, max_length=256, num_beams=5, early_stopping=True
        )

    # Move result back to CPU before decoding
    result = tokenizer.decode(beam_output[0].to("cpu"), skip_special_tokens=True)
    return result


if __name__ == "__main__":

    Bio = Biogpt(True, False, name="jpp")

    # set_seed(42)
    text = "COVID-19 is"
    output = Bio.ask(text)
    print(output)
