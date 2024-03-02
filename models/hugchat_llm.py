from hugchat import hugchat
from hugchat.login import Login
import numpy as np
from typing import Generator


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

from llm import LLM
from utils.utils import load_yaml


class HugChatLLM(LLM):
    def __init__(self, config: dict) -> None:
        """Initializes the HugChatLLM object."""
        super().__init__(local=False, loaded=False, name=None)
        self.config = config
        self.login()
        self.model = hugchat.ChatBot(cookies=self.cookies.get_dict())
        self.available_models = self.model.get_available_llm_models()

        if np.random.rand() < 1 / 4:
            self.model.delete_all_conversations()

        print("Available models on HugChat:")
        for id, display_name in enumerate(self.available_models):
            print(f"Model {id}:", display_name.displayName.split("/")[-1])

    def load_model(self, model_id: int) -> None:
        """Loads the HugChat model."""
        try:
            self.model.switch_llm(model_id)
            self.name = self.model.active_model.displayName.split("/")[-1]
            self.loaded = True
        except Exception as e:
            print("Error loading HugChat model: %s", e)

    def login(self) -> None:
        """Logs in to Hugging Face."""
        self.sign = Login(**self.config["huggingface_login"])
        try:
            self.cookies = self.sign.loadCookiesFromDir(
                MAIN_DIR_PATH + self.config["cookies_directory"]
            )
        except Exception as e:
            print("Error loading cookies: %s", e)
            self.cookies = self.sign.login()
            self.sign.saveCookiesToDir(
                MAIN_DIR_PATH + self.config["cookies_directory"])

    def ask(self, prompt: str, web_search: bool = False, new_conv: bool = True) -> str:
        """Ask the LLM a question."""
        if new_conv:
            id = self.model.new_conversation()
            self.model.change_conversation(id)
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        return str(self.model.query(prompt, web_search=web_search))

    def ask_stream(self, prompt: str, web_search: bool = False, new_conv: bool = True) -> Generator:
        """Streams the response from the LLM."""
        if new_conv:
            id = self.model.new_conversation()
            self.model.change_conversation(id)
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        for resp in self.model.query(prompt, stream=True, web_search=web_search):
            try:
                yield resp["token"]
            except Exception as e:
                print("Error streaming response: %s", e)


# Example usage of HugChatLLM
if __name__ == "__main__":

    CONFIG = load_yaml(MAIN_DIR_PATH + "./config.yaml")

    # Create an instance of HugChatLLM
    llm = HugChatLLM(CONFIG)

    # Load the model
    llm.load_model(1)

    # Ask a question
    prompt = "What is the meaning of life?"
    print("Response:", llm.ask(prompt))

    # Stream the response
    # print("Streaming response:")
    # for token in local_llm.ask_stream(prompt):
    #     print(token)
