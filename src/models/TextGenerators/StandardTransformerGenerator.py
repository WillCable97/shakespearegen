from src.models.TextGenerators.TextGenerator import TextGenerator
from src.data.TextToToken.TextToToken import TextToToken
from src.models.Transformer.Transformer import Transformer
import tensorflow as tf
import numpy as np


class StandardTransformerGenerator(TextGenerator):
    def __init__(self, input_str:str,source_model: Transformer 
                 ,context_sequencer: TextToToken, content_sequencer: TextToToken
                 , output_len: int,initializer= None):
        self.source_model = source_model
        self.context_length = source_model.context_length
        self.content_length = source_model.content_length
        self.context_sequencer = context_sequencer
        self.content_sequencer = content_sequencer
        self.input_str = input_str
        self.initializer = initializer
        self.output_len = output_len
        self.context_vector = self.create_context_vector()

    def generate_output(self):
        generated_tokens = []
        bookmark_tokens = self.content_sequencer.get_bookmark_tokens()
        content_vector = [[bookmark_tokens[0]]] if self.initializer is None else self.content_sequencer.tokenise([self.initializer])
        token_index = 0

        for i in range(self.output_len):
                content_padded = tf.keras.preprocessing.sequence.pad_sequences(content_vector, maxlen=self.content_length, padding='post')
                model_output = self.source_model([self.context_vector, content_padded])
                index_output = model_output[0][token_index]
                found_token = tf.argmax(index_output)
                #print(found_token)
                if found_token == bookmark_tokens[1]: 
                     print(f"\nTermination Reached for: {self.input_str[:15]}")
                     break
                content_vector = np.hstack((content_vector, [[found_token]]))
                
                token_index += 1
        return ''.join(self.content_sequencer.detokenise(content_vector))

    def create_context_vector(self):
        token_seq = self.context_sequencer.tokenise([self.input_str])
        token_vals = self.context_sequencer.get_bookmark_tokens()
        if self.context_sequencer.bookmark_status():
            token_seq = [[token_vals[0]] + seq + [token_vals[1]] for seq in token_seq]
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(token_seq, maxlen=self.context_length, padding='post')
        return np.array(padded_seq)
