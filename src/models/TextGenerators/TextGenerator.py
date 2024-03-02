from abc import ABC, abstractmethod

class TextGenerator(ABC):
    @abstractmethod
    def generate_output(self):
        """
            Will Generate final ouptut based on the paramters received
        """