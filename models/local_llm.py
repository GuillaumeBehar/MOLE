import os
from langchain_community.llms import LlamaCpp
from typing import Generator
from tqdm import tqdm

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))

from llm import LLM
from utils.utils import *

MAIN_DIR_PATH = up(up(os.path.abspath(__file__)))


class LocalLLM(LLM):
    def __init__(self) -> None:
        """Initializes the LocalLLM object with the given model path."""
        super().__init__(local=True, loaded=False, name=None)

    def load_model(self, model_path: str, nb_layer_offload: int = -1) -> None:
        """Loads the LlamaCpp model with the given model path."""
        try:
            self.model = LlamaCpp(
                model_path=model_path,
                n_gpu_layers=nb_layer_offload,
            )
            self.loaded = True
            self.name = get_filename(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def unload_model(self) -> None:
        """Unloads the LlamaCpp model."""
        del self.model
        self.model = None
        self.loaded = False

    def ask(self, prompt: str, web_search: bool = False) -> str:
        """Ask the LLM a question."""
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        return self.model(prompt)

    def ask_stream(self, prompt: str, web_search: bool = False) -> Generator:
        """Streams the response from the LLM."""
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        return self.model.stream(prompt)


# Example usage of LocalLLM
if __name__ == "__main__":
    config_path = MAIN_DIR_PATH + "./config.yaml"
    model_directory = load_yaml(config_path)["model_directory"]
    print(get_gguf_paths(model_directory))
    model_path = get_gguf_paths(model_directory)[1]

    # Create an instance of LocalLLM
    local_llm = LocalLLM()

    # Load the model
    local_llm.load_model(model_path)

    # Ask a question
    questions = [
        "Comment vas-tu?",
        "Pourqoi tu es triste?",
        "Quel est ton nom?",
        "Quel est ton âge?",
        "Quel est ton sexe?",
    ]

    for question in tqdm(questions, desc="Asking questions"):
        prompt = f"""
        <s> [INST] Vous êtes un assistant pour les tâches de question-réponse. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. [/INST] </s>
        [INST] Question : {question}
        Réponse : [/INST]
        """
        response = local_llm.ask(prompt)
        print("Response:", response)

    # Stream the response
    # print("Streaming response:")
    # for token in local_llm.ask_stream(prompt):
    #     print(token)
