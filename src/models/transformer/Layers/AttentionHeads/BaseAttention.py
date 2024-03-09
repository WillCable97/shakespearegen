import tensorflow as tf
import keras

class BaseAttention(tf.keras.layers.Layer):
    """
        Base Attention, only contains the layers and architecture
    """
    def __init__(self, num_heads: int, embedding_dimension: int):
        super().__init__()
        self.multi_head_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dimension)
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()
        self.supports_masking=True
