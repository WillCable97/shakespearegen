import tensorflow as tf 
from src.models.Transformer.Layers.PositionalEmbedding import PositionalEmbedding
from src.models.Transformer.Layers.Encoder.EncoderComponent import EncoderComponent

class EncoderLayer(tf.keras.layers.Layer):
    """
        Layer of chained together decoder blocks with positional embedding to start with
        Was created for a singular embedding space
    """
    def __init__(self, vocab_size: int, embedding_dimension: int, sequence_length: int
                 ,num_heads: int, dense_dimension: int, num_att_layers: int, dropout_rate=0.1):
        super().__init__()
        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size, embedding_dimension=embedding_dimension
                                                        , sequence_length=sequence_length)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense_sequence = [EncoderComponent(num_heads=num_heads, embedding_dimension=embedding_dimension
                             ,dense_dimension=dense_dimension) for i in range(num_att_layers)]
        
    def call(self, context):
        x = self.positional_embedding(context)
        x = self.dropout(x)
        
        for encoder in self.dense_sequence:
            x = encoder(x)

        return x