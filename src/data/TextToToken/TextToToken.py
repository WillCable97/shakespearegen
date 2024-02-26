from abc import ABC, abstractmethod

class TextToToken(ABC):
    """
        Generalised wrapper for handling the tokenizing of data inputs
    """

    @abstractmethod
    def init_with_input(self, input: list):
        """
            Will create tokens on original text
            Will evaluate vocab size here
        """

    @abstractmethod
    def tokenise(self, input: list) -> list:
        """
            Given another input, will convert to tokens
        """
    
    @abstractmethod
    def detokenise(self, input: list) -> list:
        """
            Will take lit of takens and convert to string
        """

    def get_vocab_size(self) -> int: 
        return self.vocab_size

    def detokenise_to_string(self, input: list) ->str:
        return ' '.join(self.detokenise(input))