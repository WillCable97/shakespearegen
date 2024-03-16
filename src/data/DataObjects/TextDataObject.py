from typing import Protocol

class TextDataObject(Protocol):
    def pad_sequences(self, padding_length: int) -> None:
        ...

    def unpad_sequance(self) -> None:
        ...

    def create_tf_dataset(self):
        ...

    def create_label(self):
        ...

    def create_val_set(self, percentage: float) -> list:
        ...

    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        ...
