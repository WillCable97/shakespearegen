import tensorflow as tf
import numpy as np

def positional_encoding_matrix(sequence_length: int, embedding_dimension: int):
    """
        Creates a matrix of alterniting frequency trig functions
        For learning positional information about words in strings
    """
    #Create embedding vector
    effective_depth = embedding_dimension / 2
    depth_vector = np.repeat(np.arange(effective_depth), 2)
    frequency_vector = 1/(10000**((2* depth_vector)/embedding_dimension))

    #Token vector
    sequence_vector = np.arange(sequence_length)

    #Create matrix
    pos_encoding = sequence_vector.reshape([-1,1]) * frequency_vector.reshape([1,-1])
    pos_encoding[:,::2] = np.sin(pos_encoding[:,::2])
    pos_encoding[:,1::2] = np.cos(pos_encoding[:,1::2])
    return tf.cast(pos_encoding, dtype=tf.float32)



class PositionalEmbedding(tf.keras.layers.Layer):
    """
        Layer responsible for placing words in embedding space
        Contains embedding layer as well as application of positional matrix
    """
    def __init__(self, vocab_size: int, embedding_dimension: int, sequence_length: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dimension, mask_zero=True)
        self.positional_matrix = positional_encoding_matrix(sequence_length=sequence_length, embedding_dimension=embedding_dimension)

    def compute_mask(self, *args, **kwargs):
        return self.embedding_layer.compute_mask(*args, **kwargs)
    
    def call(self, x):
        x = self.embedding_layer(x)
        x = x + self.positional_matrix[tf.newaxis, :, :]
        return x
