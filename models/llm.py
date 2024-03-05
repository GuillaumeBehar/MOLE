from typing import Generator


class LLM:
    def __init__(self, local: bool, loaded: bool, name: str | None) -> None:
        """Initialize the LLM class."""
        self.local = local
        self.loaded = loaded
        self.name = name
        self.model = None

    def ask(self, prompt: str, web_search: bool = False, new_conv: bool = True) -> str:
        """Ask the LLM a question."""
        pass  # This method will be implemented in subclasses

    def ask_stream(self, prompt: str, web_search: bool = False, new_conv: bool = True) -> Generator:
        """Streams the response from the LLM."""
        pass  # This method will be implemented in subclasses
