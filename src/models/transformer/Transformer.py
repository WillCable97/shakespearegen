import tensorflow as tf
from src.models.Transformer.Layers.Decoder.DecoderLayer import DecoderLayer
from src.models.Transformer.Layers.Encoder.EncoderLayer import EncoderLayer


class Transformer(tf.keras.Model):
    def __init__(self, vocab_size: int, context_vocab_size: int, embedding_dimension: int
                 , context_length: int, content_length: int,num_heads: int
                 , dense_dimension: int, num_att_layers: int, dropout_rate=0.1):
        super().__init__()  
        self.context_length = context_length
        self.content_length = content_length

        self.encoder_layer = EncoderLayer(vocab_size=context_vocab_size+1,embedding_dimension=embedding_dimension
                                          ,sequence_length=context_length,num_heads=num_heads
                                          ,dense_dimension=dense_dimension,num_att_layers=num_att_layers)
        
        self.decoder_layer = DecoderLayer(vocab_size=vocab_size+1,embedding_dimension=embedding_dimension
                                          ,sequence_length=content_length,num_heads=num_heads
                                          ,dense_dimension=dense_dimension,num_att_layers=num_att_layers)
        
        self.final_dense = tf.keras.layers.Dense(vocab_size+1)

    def call(self, inputs):
        context, content  = inputs
        context = self.encoder_layer(context=context) 
        x = self.decoder_layer(content=content, context=context)
        logits = self.final_dense(x)
        
        try:
            del logits._keras_mask
        except AttributeError:
            pass
        
        return logits
        