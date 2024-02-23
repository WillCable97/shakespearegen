#from abc import ABC, abstractmethod
from typing import Protocol
import typing

class TextDataObject(Protocol):
    def iniatiate_with_loader(self, data_loader: typing.Callable[[list], list], **kwargs):
        ...

    def fit(self, input: list):
        ...

    def map_text(self, input: list, is_initial = False) -> list: 
        ...

    def final_dataset(self, input_path: str, sequence_length: int, batch_size: int, buffer_size: int, padding_required = False):
        ...
