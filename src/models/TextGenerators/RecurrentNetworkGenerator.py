from src.models.TextGenerators.TextGenerator import TextGenerator
from src.data.TextToToken.TextToToken import TextToToken
from src.models.Transformer.Transformer import Transformer
import tensorflow as tf
import numpy as np
import keras


class RecurrentNetworkGenerator(TextGenerator):
    def __init__(self, input_str:str,source_model: keras.models.Model 
                 ,sequencer: TextToToken, output_len: int):
        self.source_model = source_model
        self.sequencer = sequencer
        self.input_str = input_str
        self.output_len = output_len
        self.source_model.build(tf.TensorShape([1, None]))
        self.source_model.reset_states()
        

    def generate_output(self):
        seq_tokens = self.sequencer.tokenise([self.input_str])
        seq_tokens = tf.convert_to_tensor(seq_tokens)
        generated_tokens = []

        for i in range(self.output_len):
             model_output = self.source_model(seq_tokens)
             model_output = tf.squeeze(model_output, 0)
             predicted_id = tf.random.categorical(model_output, num_samples=1)[-1,0].numpy()
             if predicted_id ==0: continue
             generated_tokens.append(predicted_id)
             seq_tokens = tf.concat([seq_tokens, [[predicted_id]]], axis = 1)

        return ''.join(self.sequencer.detokenise([generated_tokens]))