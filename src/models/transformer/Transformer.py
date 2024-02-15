import tensorflow as tf
from src.models.transformer.PositionalEmbedding import PositionalEmbedding
from src.models.transformer.MultiHeadAttention import CausalSelfAttention

class Transformer(tf.keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding_layer = PositionalEmbedding(vocab_size, embedding_dim)
        self.att_layer = CausalSelfAttention(num_heads = 2, key_dim = 256, dropout=0.1)
    

    def call(self, x):
        x = self.embedding_layer(x)
        x = self.att_layer(x)
        return x