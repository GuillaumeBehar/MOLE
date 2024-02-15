from hugchat import hugchat
from hugchat.login import Login
import numpy as np
from typing import Generator

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
from llm import LLM
from utils import *

MAIN_DIR_PATH = up(up(os.path.abspath(__file__)))


class HugChatLLM(LLM):
    def __init__(self, config: dict) -> None:
        """Initializes the HugChatLLM object."""
        super().__init__(local=False, loaded=False, name=None)
        self.config = config
        self.login()
        self.load_model()

    def load_model(self) -> None:
        """Loads the HugChat model."""
        try:
            self.model = hugchat.ChatBot(cookies=self.cookies.get_dict())
            self.name = self.model.active_model.displayName
            self.available_models = self.model.get_available_llm_models()

            if np.random.rand() < 1 / 4:
                self.model.delete_all_conversations()
            id = self.model.new_conversation()
            self.model.change_conversation(id)

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
            self.sign.saveCookiesToDir(MAIN_DIR_PATH + self.config["cookies_directory"])

    def ask(self, prompt: str, web_search: bool = False) -> str:
        """Ask the LLM a question."""
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        return self.model.query(prompt, web_search=web_search)

    def ask_stream(self, prompt: str, web_search: bool = False) -> Generator:
        """Streams the response from the LLM."""
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        for resp in self.model.query(prompt, stream=True, web_search=web_search):
            try:
                yield resp["token"]
            except Exception as e:
                print("Error streaming response: %s", e)


# Example usage of HugChatLLM
if __name__ == "__main__":
    config = load_yaml(MAIN_DIR_PATH + "./config.yaml")

    # Create an instance of HugChatLLM
    hugchat_llm = HugChatLLM(config)

    # Ask a question
    question = "Comment vas-tu?"
    prompt = f"""
    <s> [INST] Vous êtes un assistant pour les tâches de question-réponse. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. [/INST] </s>
    [INST] Question : {question}
    Réponse : [/INST]
    """
    response = hugchat_llm.ask(prompt)
    print("Response:", response)

    # Stream the response
    print("Streaming response:")
    for token in hugchat_llm.ask_stream(prompt):
        print(token)
