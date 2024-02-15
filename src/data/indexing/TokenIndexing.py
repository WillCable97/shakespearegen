from src.data.indexing.IndexBase import TextIndexer
from tensorflow.keras.preprocessing.text import Tokenizer

class TokenIndexing(TextIndexer):
    def __init__(self, tokenizer: Tokenizer):
        self.data_init = False
        self.tokenizer = tokenizer

    def create_mapping(self, input_text: str) -> None:
        self.tokenizer.fit_on_texts(input_text)
        text_to_int = self.tokenizer.word_index
        int_to_text = self.tokenizer.index_word
        self.mapped_text = self.tokenizer.texts_to_sequences(input_text)
        self.text_to_int = text_to_int
        self.int_to_text = int_to_text
        self.data_init = True
        
    def text_to_int(self, input_str: str) -> int:
        if not self.data_init: return -1 #This should be done with decorators !!!!
        return self.text_to_int[input_str]
    
    def int_to_text(self, input_token: int) -> str:
        if not self.data_init: return ""
        return self.int_to_text[input_token]
    
    def mapped_input(self) -> list:
        if not self.data_init: return []
        return self.mapped_text
    
    def token_maps(self) -> dict:
        if not self.data_init: return {}
        return self.int_to_text
    
    def reverse_token_maps(self) -> dict:
        if not self.data_init: return {}
        return self.text_to_int