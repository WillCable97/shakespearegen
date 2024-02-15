from abc import ABC, abstractmethod

class TextIndexer(ABC):
    @abstractmethod
    def create_mapping(self, input_text: str) -> None:
        """Create and initializer mapping"""

    @abstractmethod
    def text_to_int(self, input_str: str) -> int:
        """Mapping text to int"""

    @abstractmethod
    def int_to_text(self, input_token: int) -> str:
        """Mapping int to text"""

    @abstractmethod
    def mapped_input(self) -> list:
        """Return the full mapped text given as input"""

    @abstractmethod
    def token_maps(self) -> dict:
        """Return list of all tokens and corresponsing string"""

    @abstractmethod
    def reverse_token_maps(self) ->dict:
        """Return list of strings and corresponding tokens"""

