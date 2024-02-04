import tensorflow as tf
from datasets import load_dataset
from dataclasses import dataclass, field 
from abc import ABC, abstractmethod

#******************************INDEXERS******************************#
class TextIndexer(ABC):
    @abstractmethod
    def create_maps(self, input_text: str):
        """Create text maps"""

    #Getter functions
    def ret_text_to_int(self) -> dict: return self.text_to_int
    
    def ret_int_to_text(self) -> dict: return self.int_to_text
    
    def ret_mapped_text(self) -> list: return self.mapped_text

class CharacterIndexing(TextIndexer):
    def create_maps(self, input_text: str):
        ordered_text = sorted(set(input_text))
        text_to_int = {c: i for i,c in enumerate(ordered_text)}
        int_to_text = {text_to_int[c] : c for c in text_to_int}
        self.mapped_text = [text_to_int[c] for c in input_text]
        self.text_to_int = text_to_int
        self.int_to_text = int_to_text


@dataclass
class DataObject:
    hugging_face_name: str
    index_obj: TextIndexer
    sequence_length : int
    batch_size: int
    buffer_size: int
    all_text: str = field(init=False)
    final_dataset: tf.data.Dataset = field(init=False)

    def __post_init__(self):
        hg_data = load_dataset(self.hugging_face_name)
        self.all_text = hg_data["train"]["text"][0]
        self.index_obj.create_maps(self.all_text)
        self.create_final_dataset()

    def text_to_int(self) -> dict: 
        return self.index_obj.ret_text_to_int()

    def int_to_text(self) -> dict:
        return self.index_obj.ret_int_to_text()

    def offset_sequnce(self, input_sequence:str):
        #Function for creating target variables
        return input_sequence[:-1], input_sequence[1:]
    
    def create_final_dataset(self):
        mapped_text = self.index_obj.ret_mapped_text()
        #print(mapped_text)
        #Creating formal tensorflow objects
        tf_data_obj = tf.data.Dataset.from_tensor_slices(mapped_text)
        tf_data_obj_sequenced = tf_data_obj.batch(self.sequence_length, drop_remainder=True) #Creates sequnces of lenght 100
        raw_tf_data = tf_data_obj_sequenced.map(self.offset_sequnce)

        

        #Batch and shuffle for final data
        self.final_dataset = raw_tf_data.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)


char_indexing_obj = CharacterIndexing()
data_object_initial = DataObject(hugging_face_name="tiny_shakespeare"
                                 ,index_obj=char_indexing_obj
                                 ,sequence_length=100
                                 ,batch_size=64
                                 ,buffer_size=200)



