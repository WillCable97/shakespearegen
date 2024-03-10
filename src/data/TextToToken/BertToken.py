from src.data.TextToToken.TextToToken import TextToToken
from transformers import  AutoTokenizer


class BertToken(TextToToken):
    def __init__(self):
        self.b_token = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
        self.bookmarked = False
        self.bookmark_tokens = [0,0]
        

    def init_with_input(self, input: list):
        pass

    def tokenise(self, input: list):
        full_str = ' '.join(input)
        split_list = full_str.split(' ')
        unique_words = set(split_list)
        self.vocab_size = len(unique_words)
        retlist = [self.b_token.encode(a) for a in input]
        return retlist
    
    def detokenise(self, input: list) -> list:
        return self.b_token.decode(input)

