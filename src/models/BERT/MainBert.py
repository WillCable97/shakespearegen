import keras
from transformers import TFBertModel


class BertModel(keras.Model):
    def __init__(self, vocab_size, context_length, content_length):
        super().__init__()
        self.bert_model = TFBertModel.from_pretrained("google-bert/bert-base-uncased")
        self.dense_comp = keras.layers.Dense(vocab_size, activation='softmax')
        self.context_length = context_length
        self.content_length = content_length
    def call(self, inputs):
        inputs = self.bert_model(inputs)
        pooled_output = inputs.last_hidden_state
        x = self.dense_comp(pooled_output)
        return x 