import tensorflow as tf 
from src.models.Transformer.Layers.PositionalEmbedding import PositionalEmbedding
from src.models.Transformer.Layers.SingleEmbeddedDecoderComponent import SingleEmbeddedDecoderComponent


class SingleEmbeddingDecoderLayer(tf.keras.layers.Layer):
    """
        Layer of chained together decoder blocks with positional embedding to start with
        Was created for a singular embedding space
    """
    def __init__(self, vocab_size: int, embedding_dimension: int, sequence_length: int
                 ,num_heads: int, dense_dimension: int, num_att_layers: int, dropout_rate=0.1):
        super(SingleEmbeddingDecoderLayer, self).__init__()


        self.embedding_dimension = embedding_dimension
        self.num_att_layers = num_att_layers

        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size, embedding_dimension=embedding_dimension
                                                        , sequence_length=sequence_length)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense_sequence = [SingleEmbeddedDecoderComponent(num_heads=num_heads, embedding_dimension=embedding_dimension
                             ,dense_dimension=dense_dimension) for i in range(num_att_layers)]
        
    def call(self, x):
        x = self.positional_embedding(x)
        x = self.dropout(x)
        
        for decoder in self.dense_sequence:
            x = decoder(x)

        return x
