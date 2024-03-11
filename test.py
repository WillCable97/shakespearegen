

import tensorflow as tf
from keras.models import clone_model

class LSTM_model(tf.keras.Model):
    def __init__(self, vocab_size:int, embedding_dim: int, rnn_units:int, batch_size:int):
        super().__init__()
        self.emb_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm_layer = tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')
        self.dense_comp = tf.keras.layers.Dense(vocab_size)
    def build(self, input_shape):
        super(LSTM_model, self).build(input_shape)
        
    def call(self, input):
        input = self.emb_layer(input)
        input = self.lstm_layer(input)
        input = self.dense_comp(input)
        return input

# Create an instance of the model
lstm_gen_inst = LSTM_model(vocab_size=65 + 1, embedding_dim=256,
                            rnn_units=256, batch_size=1)

# Build the model
lstm_gen_inst.build(tf.TensorShape([1, None]))

# Check if the model is built successfully
print(lstm_gen_inst.get_weights())
