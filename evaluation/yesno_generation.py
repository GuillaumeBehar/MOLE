import argparse
import os
from groq import Groq
import json

def main():
    parser = argparse.ArgumentParser(description="Script Groq")
    parser.add_argument("--api-key", help="Clé API Groq")
    parser.add_argument("--json-file", help="fichier json contenant les réponses générées")
    args = parser.parse_args()

    if args.api_key:
        api_key = args.api_key
    else:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Veuillez spécifier la clé API Groq en tant qu'argument --api-key ou définir la variable d'environnement GROQ_API_KEY.")
        return

    client = Groq(api_key=api_key)
    with open(args.json_file, 'r') as json_file:
        input_dict = json.load(json_file)
    output_dict = generate_yesno_from_mixtral(client, input_dict, "target_answer")

    with open('mixtral_decisions.json', 'w') as json_file:
        json.dump(output_dict, json_file, indent=4)

def generate_yesno_from_mixtral(client, generated_dict: dict, value_to_evaluate: str) -> dict:
    format_prompt = "You are answering a boolean question. Important: return 'yes' if the answer is positive and 'no' if the answer is negative."
    one_shot_prompting = ("Example: Are paraproteins a common cause of interferences with automated chemistry methods ? Due to "
                          "the fact that We demonstrated that paraprotein interferences with TBIL, DBIL, "
                          "and HDL-C are relatively common and provided explanations why these interferences "
                          "occurred. Although it is difficult to predict which specimens cause interferences, "
                          "spurious results appeared method and concentration dependent. The answer is yes.")
    begin_prompt = 'Question: '
    mid_prompt = 'Due to the fact that '
    end_prompt = 'The answer is :'

    for key, value in generated_dict.items():
        question = value["question"]
        long_answer = value[value_to_evaluate]
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": format_prompt + one_shot_prompting + begin_prompt + question + mid_prompt + long_answer + end_prompt,
                }
            ],
            model="mixtral-8x7b-32768",
            max_tokens=3,
        )
        value["short_generated"] = chat_completion.choices[0].message.content
        print(value["short_generated"])
    return generated_dict

if __name__ == "__main__":
    main()
