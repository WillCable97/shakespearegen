import keras


class DecoderLayer(keras.layers.Layer):
    def __init__(self, dense_dimension, embedding_dimension, num_layers, content_vocab, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.embedding = keras.layers.Embedding(content_vocab, embedding_dimension)
        self.decoder_layers = [keras.layers.LSTM(dense_dimension, return_sequences=True, return_state=True) for _ in range(num_layers)]
        self.dense = keras.layers.Dense(content_vocab)

    def call(self, inputs, encoder_states):
        x = self.embedding(inputs)
        for layer in self.decoder_layers:
            x, _, _ = layer(x, initial_state=encoder_states[-2:])
        outputs = self.dense(x)
        return outputs