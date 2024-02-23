import tensorflow as tf
from src.models.Transformer.Layers.DecoderLayer import DecoderLayer
from src.models.Transformer.Layers.EncoderLayer import EncoderLayer


class Transformer(tf.keras.Model):
    def __init__(self, vocab_size: int, context_vocab_size: int, embedding_dimension: int, sequence_length: int
                 ,num_heads: int, dense_dimension: int, num_att_layers: int, dropout_rate=0.1):
        super().__init__()  

        self.decoder_layer = DecoderLayer(vocab_size=vocab_size,embedding_dimension=embedding_dimension
                                          ,sequence_length=sequence_length,num_heads=num_heads
                                          ,dense_dimension=dense_dimension,num_att_layers=num_att_layers)
        
        self.encoder_layer = EncoderLayer(vocab_size=context_vocab_size,embedding_dimension=embedding_dimension
                                          ,sequence_length=sequence_length,num_heads=num_heads
                                          ,dense_dimension=dense_dimension,num_att_layers=num_att_layers)
        self.final_dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        context, x  = inputs
        context = self.encoder_layer(context) 
        x = self.decoder_layer(x, context)
        logits = self.final_dense(x)
        
        try:
            del logits._keras_mask
        except AttributeError:
            pass
        
        return logits
        