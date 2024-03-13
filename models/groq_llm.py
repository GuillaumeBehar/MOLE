import time
import numpy as np
from typing import Generator, Dict
from typing import Any
from groq import Groq
import os


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

from models.llm import LLM
from utils.custom_utils import load_yaml


class GroqLLM(LLM):
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the GroqLLM object."""
        super().__init__(local=False, loaded=False, name=None)
        self.config = config
        self.client = Groq(api_key=config["groq"]["api_key"])
        self.available_models = self.get_available_models()

        print("Available models on Groq:")
        for id, display_name in enumerate(self.available_models):
            print(f"Model {id}:", display_name)

    def load_model(self, model_id: int) -> None:
        """Loads the Groq model."""
        self.model = self.client.chat.completions.create(
            model=self.available_models[model_id],
            messages=[
                {
                    "role": "system",
                    "content": "you are a helpful assistant."
                }
            ]
        )
        self.name = self.available_models[model_id]
        self.loaded = True

    def get_available_models(self) -> list[str]:
        """Returns the list of available models on Groq."""
        response = self.client.models.list()
        return [model.id for model in response.data]

    def ask(self, prompt: str, web_search: bool = False) -> str:
        """Ask the LLM a question."""
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        attempt = 0
        while attempt < 5:
            try:
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                attempt += 1
                print("Error asking question: %s", repr(e))
                print("Waiting 1 minutes before retrying.")
                time.sleep(60)

    def ask_stream(self, prompt: str, web_search: bool = False) -> Generator[str, None, None]:
        """Streams the response from the LLM."""
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=True
            )
            for choice in response.choices:
                yield choice.message.content
        except Exception as e:
            print("Error streaming response: %s", e)


# Example usage of GroqChatLLM
if __name__ == "__main__":

    CONFIG = load_yaml(MAIN_DIR_PATH + "./config.yaml")

    # Create an instance of GroqChatLLM
    llm = GroqLLM(CONFIG)

    # Load the model
    llm.load_model(1)

    # Ask a question
    prompt = "What is the meaning of life?"
    print("Response:", llm.ask(prompt))

    # Stream the response
    # print("Streaming response:")
    # for token in local_llm.ask_stream(prompt):
    #     print(token)
