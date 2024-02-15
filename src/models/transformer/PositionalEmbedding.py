import numpy as np
import tensorflow as tf


def positional_encoding(length: int, depth: int):
    effective_depth = depth / 2
    depth_vector = np.repeat(np.arange(effective_depth), 2)
    frequency_vector = 1/(10000**((2* depth_vector)/depth))
    sequence_vector = np.arange(length)
    pos_encoding = sequence_vector.reshape([-1,1]) * frequency_vector.reshape([1,-1])
    pos_encoding[:,::2] = np.sin(pos_encoding[:,::2])
    pos_encoding[:,1::2] = np.cos(pos_encoding[:,1::2])
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, depth):
        super().__init__()
        self.depth = depth
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, depth, mask_zero=True)
        self.positional_func = positional_encoding(vocab_size, depth)
    
    def call(self, x):
        length = tf.shape(x)[1] #this is for baches
        x = self.embedding_layer(x)
        x *= tf.math.sqrt(tf.cast(self.depth, tf.float32)) #not sure
        x = x + self.positional_func[tf.newaxis, :length, :]
        return x