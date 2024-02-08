import os
from typing import List
from langchain_community.llms import LlamaCpp

MODEL_DIRECTORY = r"C:\Users\cranc\Documents\Models"


def get_gguf_paths(directory: str = MODEL_DIRECTORY) -> List[str]:
    """Get paths of all files with '.gguf' extension in the specified directory and its subdirectories."""
    gguf_paths = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(directory)
        for name in files
        if name.endswith(".gguf")
    ]
    return gguf_paths


def get_filename(path: str) -> str:
    """Get the filename from the given path."""
    return os.path.basename(path)


class LocalLLM:
    def __init__(self, model_path: str) -> None:
        """Initializes the LlamaCpp object with the given model path."""
        self.model = LlamaCpp(
            model_path=model_path, n_gpu_layers=20, n_batch=512, n_ctx=2048
        )
