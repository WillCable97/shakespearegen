from dataclasses import dataclass, field
import tensorflow as tf
from src.data.indexing.IndexBase import TextIndexer
from datasets import load_dataset
from src.data.data_objs.DataObject import DataObject
import numpy as np

def pad_list(input_list, max_size):
    list_size = len(input_list)
    if list_size >= max_size: return input_list[:max_size]
    padding_count = max_size - list_size
    padding_list = [0] * padding_count 
    return input_list + padding_list

@dataclass
class TextDataObject(DataObject):
    index_obj: TextIndexer
    sequence_length: int
    batch_size: int
    buffer_size: int
    all_text: list = field(init=False)
    token_text: list = field(init=False)
    inputs: list = field(init=False)
    outputs: list = field(init=False)
    final_dataset: tf.data.Dataset = field(init=False)
    data_loaded = False

    #load data -> create tokens -> structure data -> create training data -> create final tf dataset obj



    def load_from_file(self, file_path: str):#defaults to split by line
        text_file = open(file_path, 'r')
        all_text = text_file.readlines()
        self.all_text = all_text
        text_file.close()
        self.data_loaded = True

    def load_from_hugging_face(self, hugging_face_name: str):#Default it what the hugging face data structure is
        hg_data = load_dataset(hugging_face_name)
        self.all_text = [hg_data["train"]["text"]]
        self.data_loaded = True

    def create_tokens(self):
        self.index_obj.create_mapping(self.all_text)
        self.token_text = self.index_obj.mapped_input()

    def add_padding(self):
        self.token_text = [pad_list(x, self.sequence_length) for x in self.token_text]
    
    def create_sequences(self):
        flat_t_list = self.flat_list(self.token_text)
        sequence_length = self.sequence_length
        number_of_sequences = len(self.token_text) // sequence_length
        self.token_text = [flat_t_list[i* sequence_length: (i+1) * sequence_length] for i in range(number_of_sequences)]

    def flat_list(self, input_list: list) -> list: 
        return [string_f for seq in input_list for string_f in seq]
    
    def training_data_set_recursive(self):
        input = [seq[:-1] for seq in self.token_text]
        ouptut = [seq[1:] for seq in self.token_text]
        self.inputs = np.array(input)
        self.outputs = np.array(ouptut)

    def create_final_dataset(self):
        input_to_tensor = tf.convert_to_tensor(self.inputs) 
        labels_to_tensor = tf.convert_to_tensor(self.outputs)
        input_dataset = tf.data.Dataset.from_tensor_slices(input_to_tensor)
        label_dataset = tf.data.Dataset.from_tensor_slices(labels_to_tensor)
        tf_main_dataset = tf.data.Dataset.zip((input_dataset, label_dataset))
        batched_dataset = tf_main_dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)
        self.final_dataset = batched_dataset

    def vocab_size(self) -> int:
        return len(self.index_obj.token_maps())