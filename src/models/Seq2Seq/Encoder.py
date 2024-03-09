import keras

class EncoderLayer(keras.layers.Layer):
    def __init__(self, dense_dimension, embedding_dimension, context_vocab, num_layers, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.embedding = keras.layers.Embedding(context_vocab, embedding_dimension)
        self.encoder_layers = [keras.layers.LSTM(dense_dimension, return_sequences=True, return_state=True) 
                               for _ in range(num_layers)]

    def call(self, inputs):
        x = self.embedding(inputs)
        states = []
        for layer in self.encoder_layers:
            x, state_h, state_c = layer(x)
            states += [state_h, state_c]
        return x, states