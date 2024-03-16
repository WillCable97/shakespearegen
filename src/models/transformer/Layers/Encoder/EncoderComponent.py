import tensorflow as tf
from src.models.Transformer.Layers.AttentionHeads.GlobalSelfAttention import GlobalSelfAttention
from src.models.Transformer.Layers.DenseComponent import DenseComponent


class EncoderComponent(tf.keras.layers.Layer):
    """
        A single Decoding layer, these are chained together in succesion to create the complete Decoder
        Consists onl of an attention block and the following feed forward network
    """
    def __init__(self, num_heads: int, embedding_dimension: int, dense_dimension: int):
        super().__init__()
        self.attention_block_layer = GlobalSelfAttention(num_heads=num_heads, embedding_dimension=embedding_dimension)
        self.dense_component = DenseComponent(embedding_dimension=embedding_dimension, dense_dimension=dense_dimension)

    def call(self, x):
        x = self.attention_block_layer(x)
        x = self.dense_component(x)
        return x
