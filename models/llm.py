from typing import Generator


class LLM:
    def __init__(self, local: bool, loaded: bool, name: str) -> None:
        """Initialize the LLM class."""
        self.local = local
        self.loaded = loaded
        self.name = name
        self.model = None

    def ask(self, question: str) -> str:
        """Ask the LLM a question."""
        pass  # This method will be implemented in subclasses

    def ask_stream(self, prompt: str, web_search: bool = False) -> Generator:
        """Streams the response from the LLM."""
        pass  # This method will be implemented in subclasses
