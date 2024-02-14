from typing import Generator


class LLM:
    def __init__(self, local: bool, loaded: bool, name: str) -> None:
        self.local = local  # Indicates whether the LLM is local or remote
        self.loaded = loaded  # Indicates whether the model is loaded
        self.name = name  # Name of the LLM
        self.model = None  # The actual LLM model

    def ask(self, question: str) -> str:
        """Ask the LLM a question."""
        pass  # This method will be implemented in subclasses

    def ask_stream(self, prompt: str, web_search: bool = False) -> Generator:
        """Streams the response from the HugChat model."""
        pass  # This method will be implemented in subclasses
