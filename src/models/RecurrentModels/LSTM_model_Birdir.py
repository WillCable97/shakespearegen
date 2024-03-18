import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding, LSTM, Dense, BatchNormalization, Bidirectional
from tensorflow.keras import regularizers

class LSTM_model_Birdir(Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size,regularizer, dropout_rate):
        super().__init__()
        self.emb_layer = Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None])
        #self.lstm_layer = LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')
        self.lstm_layer = Bidirectional(LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'
                               , kernel_regularizer=regularizers.l2(regularizer), dropout=dropout_rate))
        self.batch_norm = BatchNormalization()
        self.dense_comp = Dense(vocab_size)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, input):
        input = self.emb_layer(input)
        input = self.lstm_layer(input)
        input = self.batch_norm(input)
        input = self.dense_comp(input)
        return input
    
"""

    def build(self, input_shape):
        super(LSTM_model, self).build(input_shape)




# Example usage
vocab_size_shake = 10000
embedding_dimension = 128
dense_dimension = 256

lstm_gen_inst = LSTM_model(vocab_size=vocab_size_shake + 1, embedding_dim=embedding_dimension,
                            rnn_units=dense_dimension, batch_size=1)

# Build the model with a specific input shape
lstm_gen_inst.build(tf.TensorShape([1, None]))

# Check the model architecture
lstm_gen_inst.summary()
"""