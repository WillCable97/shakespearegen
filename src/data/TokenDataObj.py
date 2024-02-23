import typing
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from src.data.TextDataObject import TextDataObject


def create_offset_labels(input_tensor):
    return input_tensor[:-1], input_tensor[1:]

class TokenDataObj(TextDataObject):
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        
    def iniatiate_with_loader(self, data_loader: typing.Callable[[list], list], **kwargs):
        self.raw_data = data_loader(**kwargs)
        self.fit(self.raw_data)

    def fit(self, input: list):
        self.tokenizer.fit_on_texts(input)
        self.map_text(input=input, is_initial=True)
        self.vocab_size = len(self.tokenizer.index_word)

    def map_text(self, input: list, is_initial = False) -> list: 
        mapped_seq = self.tokenizer.texts_to_sequences(input)
        if is_initial: self.mapped_seq = mapped_seq
        return mapped_seq

    def final_dataset(self, sequence_length: int, batch_size: int, buffer_size: int
                      , padding_required = False, create_label=True, batching_required = True):
        if padding_required: self.mapped_seq = tf.keras.preprocessing.sequence.pad_sequences(self.mapped_seq, maxlen=sequence_length+1, padding="post")
        tensor_slices = tf.data.Dataset.from_tensor_slices(self.mapped_seq)
        if create_label: tensor_slices = tensor_slices.map(create_offset_labels)
        if batching_required: tensor_slices = tensor_slices.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        self.final_dataset_obj = tensor_slices
        return tensor_slices

