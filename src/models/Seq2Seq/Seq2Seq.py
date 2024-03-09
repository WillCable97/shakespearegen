from src.models.Seq2Seq.Encoder import EncoderLayer
from src.models.Seq2Seq.Decoder import DecoderLayer
import keras

class Seq2SeqModel(keras.Model):
    def __init__(self, dense_dimension, embedding_dimension, num_layers, content_vocab,context_vocab
                 ,context_length, content_length, **kwargs):
        super(Seq2SeqModel, self).__init__(**kwargs)
        self.encoder = EncoderLayer(dense_dimension=dense_dimension, embedding_dimension=embedding_dimension
                                    ,context_vocab=context_vocab+1, num_layers=num_layers)
        
        self.decoder = DecoderLayer(dense_dimension=dense_dimension, embedding_dimension=embedding_dimension
                                    ,num_layers=num_layers, content_vocab=content_vocab+1)
        
        self.context_length = context_length
        self.content_length = content_length

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        _, encoder_states = self.encoder(encoder_inputs)
        decoder_outputs = self.decoder(decoder_inputs, encoder_states)
        return decoder_outputs