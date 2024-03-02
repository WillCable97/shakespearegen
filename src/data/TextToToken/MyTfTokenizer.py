from src.data.TextToToken.TextToToken import TextToToken
from keras.preprocessing.text import Tokenizer

class MyTfToken(TextToToken):
    """
        Wrapper for Tensorflow Tokenizer
    """
    def __init__(self, use_bookmark: bool = False):
        self.tf_tokenizer = Tokenizer()
        self.bookmarked = use_bookmark

    def init_with_input(self, input: list):
        self.tf_tokenizer.fit_on_texts(input)
        self.vocab_size = len(self.tf_tokenizer.word_index)
        self.bookmark_tokens = [self.vocab_size + 1, self.vocab_size + 2] 

        #so it doesn't break for the detokenise
        self.tf_tokenizer.index_word[self.vocab_size+1]=''
        self.tf_tokenizer.index_word[self.vocab_size+2]=''
        if self.bookmarked: self.vocab_size += 2 #this is to account for the extra 2 tokens on either side

    def tokenise(self, input: list) ->list:
        return self.tf_tokenizer.texts_to_sequences(input)
    
    def detokenise(self, input: list) -> list:
        ret_list = [f"{self.tf_tokenizer.index_word[token]} " for sequence in input for token in sequence]
        return ret_list

