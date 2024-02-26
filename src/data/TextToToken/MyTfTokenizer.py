from src.data.TextToToken.TextToToken import TextToToken
from keras.preprocessing.text import Tokenizer

class MyTfToken(TextToToken):
    """
        Wrapper for Tensorflow Tokenizer
    """
    def __init__(self):
        self.tf_tokenizer = Tokenizer()

    def init_with_input(self, input: list):
        self.tf_tokenizer.fit_on_texts(input)
        self.vocab_size = len(self.tf_tokenizer.word_index)

    def tokenise(self, input: list) ->list:
        return self.tf_tokenizer.texts_to_sequences(input)
    
    def detokenise(self, input: list) -> list:
        ret_list = [self.tf_tokenizer.index_word[token] for sequence in input for token in sequence]
        return ret_list

