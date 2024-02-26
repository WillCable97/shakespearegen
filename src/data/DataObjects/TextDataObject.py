import typing
import tensorflow as tf
from typing import Protocol
from src.data.TextToToken.TextToToken import TextToToken


def create_offset_labels(input_tensor):
    return input_tensor[:-1], input_tensor[1:]


def reorder_transformer_dataset(context_tensor, content_tensor):
   a = context_tensor
   (b, c) = content_tensor
   return (a,b) , c




class TextDataObject(Protocol):
    def pad_sequences(self, padding_length: int) -> None:
        ...

    def unpad_sequance(self) -> None:
        ...

    def create_tf_dataset(self):
        ...

    def create_label(self):
        ...

    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        ...




class TransformerTextDataObject:
    def __init__(self, context_sequencer: TextToToken, content_sequencer: TextToToken
                 , context_len: int, content_len: int, data_loader: typing.Callable[[typing.Any], list]
                 ,use_bookmark_tokens = False,**kwargs):
        #Raw data and sequencing
        self.context_sequencer = context_sequencer
        self.content_sequencer = content_sequencer
        self.context_len = context_len
        self.content_len = content_len + 1 #One will be lost when the map is applied to create labels
        self.raw_context, self.raw_content = data_loader(**kwargs)
        if use_bookmark_tokens: self.add_bookmark_tokens()
        self.context_sequencer.init_with_input(self.raw_context)
        self.content_sequencer.init_with_input(self.raw_content)
        self.token_context = self.context_sequencer.tokenise(self.raw_context)
        self.token_content = self.content_sequencer.tokenise(self.raw_content)

        #Vocabs
        self.context_vocab = self.context_sequencer.get_vocab_size()
        self.content_vocab = self.content_sequencer.get_vocab_size()

        #Create data by defaul
        self.pad_sequences()
        self.create_tf_dataset()
        self.create_label()


    def add_bookmark_tokens(self):
        self.raw_context = ["starttoken " + x + " endtoken" for x in self.raw_context]
        self.raw_content = ["starttoken " + x + " endtoken" for x in self.raw_content]
        
    def pad_sequences(self) -> None:
        self.token_context = tf.keras.preprocessing.sequence.pad_sequences(self.token_context, maxlen=self.context_len
                                                                           , padding="post")
        self.token_content = tf.keras.preprocessing.sequence.pad_sequences(self.token_content, maxlen=self.content_len
                                                                           , padding="post")

    def unpad_sequance(self) -> None:
        pass

    def create_tf_dataset(self):
        self.context_tf_dataset = tf.data.Dataset.from_tensor_slices(self.token_context)
        self.content_tf_dataset = tf.data.Dataset.from_tensor_slices(self.token_content)

    def create_label(self):
        self.content_tf_dataset = self.content_tf_dataset.map(create_offset_labels)

    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        preliminary_tensor = tf.data.Dataset.zip((self.context_tf_dataset, self.content_tf_dataset))
        preliminary_tensor = preliminary_tensor.map(reorder_transformer_dataset)
        self.final_tf_dataset = preliminary_tensor.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return self.final_tf_dataset


class StandardTextDataObject:
    def __init__(self, text_sequencer: TextToToken, data_loader: typing.Callable[[typing.Any], list]
                 ,sequence_lenth: int, **kwargs):
        #Work with Raw Dataset
        self.text_sequencer = text_sequencer
        self.raw_data = data_loader(**kwargs)
        self.text_sequencer.init_with_input(self.raw_data)
        self.sequence_lenth = sequence_lenth
        #Work with token list
        self.token_list = self.text_sequencer.tokenise(self.raw_data)
        self.vocab_size = self.text_sequencer.get_vocab_size()

    def pad_sequences(self, padding_length: int) -> None:
        self.token_text = tf.keras.preprocessing.sequence.pad_sequences(self.token_list, maxlen=padding_length, padding="post")

    def unpad_sequance(self) -> None:
        pass

    def create_tf_dataset(self):
        self.tf_dataset = tf.data.Dataset.from_tensor_slices(self.token_list)

    def create_label(self):
        self.tf_dataset = self.tf_dataset.map(create_offset_labels)

    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        self.tf_dataset = self.tf_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return self.tf_dataset


class E2EStandardTextObject(StandardTextDataObject):
    """
        Just implements all operations that will be standatd for RNN single embedded operations
    """
    def __init__(self, text_sequencer: TextToToken, data_loader: typing.Callable[[typing.Any], list], sequence_lenth: int, **kwargs):
        super().__init__(text_sequencer, data_loader, sequence_lenth, **kwargs)
        self.pad_sequences(self.sequence_lenth)
        self.create_tf_dataset()
        self.create_label()