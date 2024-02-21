import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed


def generate_from_biogpt(sentence: str, tokenizer: BioGptTokenizer, model: BioGptForCausalLM) -> str:
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        beam_output = model.generate(**inputs,
                                     min_length=100,
                                     max_length=256,
                                     num_beams=5,
                                     early_stopping=True
                                     )
    result = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    return result


if __name__ == "__main__":
    Tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    Model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

    # set_seed(42)
    text = "COVID-19 is"
    output = generate_from_biogpt(sentence=text, tokenizer=Tokenizer, model=Model)
    print(output)
