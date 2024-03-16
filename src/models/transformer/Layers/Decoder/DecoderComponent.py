import tensorflow as tf
from src.models.Transformer.Layers.AttentionHeads.SelfAttentionHead import SelfAttentionHead
from src.models.Transformer.Layers.AttentionHeads.CrossAttention import CrossAttention
from src.models.Transformer.Layers.DenseComponent import DenseComponent

class DecoderComponent(tf.keras.layers.Layer):
    """
        A single Decoding layer, these are chained together in succesion to create the complete Decoder
        Consists onl of an attention block and the following feed forward network
    """
    def __init__(self, num_heads: int, embedding_dimension: int, dense_dimension: int):
        super().__init__()
        self.attention_block_layer = SelfAttentionHead(num_heads=num_heads, embedding_dimension=embedding_dimension)
        self.cross_attention = CrossAttention(num_heads=num_heads, embedding_dimension=embedding_dimension)
        self.dense_component = DenseComponent(embedding_dimension=embedding_dimension, dense_dimension=dense_dimension)

    def call(self, x, context):
        x = self.attention_block_layer(x)
        x = self.cross_attention(x=x, context=context)
        x = self.dense_component(x)
        return x


