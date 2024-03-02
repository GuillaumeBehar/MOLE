
from typing import Generator
from langchain_community.llms import LlamaCpp


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

from utils.utils import get_filename, load_yaml, get_extensions_paths
from models.llm import LLM


class LocalGgufLLM(LLM):
    def __init__(self, config: dict) -> None:
        """Initializes the LocalLLM object with the given model path."""
        super().__init__(local=True, loaded=False, name=None)
        self.gguf_paths = get_extensions_paths(
            config["model_directory"], "gguf")
        print("Available models on the PC:")
        for i, path in enumerate(self.gguf_paths):
            print("Model:", i, path)

    def load_model(self, model_id: int, nb_layer_offload: int = -1) -> None:
        """Loads the LlamaCpp model with the given model path."""
        try:
            model_path = self.gguf_paths[model_id]
            self.model = LlamaCpp(model_path=model_path,
                                  n_gpu_layers=nb_layer_offload)
            self.name = get_filename(model_path)
            self.loaded = True
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def unload_model(self) -> None:
        """Unloads the LlamaCpp model."""
        del self.model
        self.model = None
        self.loaded = False
        self.name = None

    def ask(self, prompt: str, web_search: bool = False, new_conv: bool = True) -> str:
        """Ask the LLM a question."""
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        return self.model(prompt)

    def ask_stream(self, prompt: str, web_search: bool = False, new_conv: bool = True) -> Generator:
        """Streams the response from the LLM."""
        if not self.loaded:
            raise ValueError("Model not loaded. Please load the model first.")
        return self.model.stream(prompt)


# Example usage of LocalLLM
if __name__ == "__main__":

    CONFIG = load_yaml(MAIN_DIR_PATH + "./config.yaml")

    # Create an instance of LocalGgufLLM
    llm = LocalGgufLLM(CONFIG)

    # Load the model
    llm.load_model(1)

    # Ask a question
    prompt = "What is the meaning of life?"
    print("Response:", llm.ask(prompt))

    # Stream the response
    # print("Streaming response:")
    # for token in local_llm.ask_stream(prompt):
    #     print(token)
