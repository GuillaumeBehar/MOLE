import os
from typing import List
from langchain_community.llms import LlamaCpp

import sys
import os
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
from llm import LLM
from utils import *

MAIN_DIR_PATH = up(up(os.path.abspath(__file__)))


def get_gguf_paths(directory: str) -> List[str]:
    """Get paths of all files with '.gguf' extension in the specified directory and its subdirectories."""
    gguf_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(directory)
        for name in files
        if name.endswith(".gguf")
    ]
    return gguf_paths


def get_filename(path: str) -> str:
    """Get the filename from the given path."""
    return os.path.basename(path)


class LocalLLM(LLM):
    def __init__(self) -> None:
        """Initializes the LocalLLM object with the given model path."""
        super().__init__(local=True, loaded=False, name=None)

    def load_model(self, model_path: str) -> None:
        """Loads the LlamaCpp model with the given model path."""
        self.model = LlamaCpp(
            model_path=model_path, n_gpu_layers=-1, n_batch=512, n_ctx=2048
        )
        self.loaded = True  # Update loaded attribute after successful loading
        self.name = get_filename(
            model_path
        )  # Update name attribute after successful loading

    def kill_model(self) -> str:
        """Kill the LlamaCpp model."""
        del self.model
        self.model = None
        self.loaded = False

    def ask(self, prompt: str) -> str:
        """Ask the LLM a question."""
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        return self.model(prompt)


# Example usage of LocalLLM
if __name__ == "__main__":
    config_path = MAIN_DIR_PATH + "./config.yaml"
    model_directory = load_yaml(config_path)["model_directory"]
    model_path = get_gguf_paths(model_directory)[0]

    # Create an instance of LocalLLM
    local_llm = LocalLLM()

    # Load the model
    local_llm.load_model(model_path)

    # Ask a question
    question = "Comment vas-tu?"
    prompt = f"""
    <s> [INST] Vous êtes un assistant pour les tâches de question-réponse. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. [/INST] </s>
    [INST] Question : {question}
    Réponse : [/INST]
    """
    response = local_llm.ask(prompt)
    print(response)

    # Kill the model
    local_llm.kill_model()
