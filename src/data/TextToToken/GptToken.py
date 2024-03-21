from src.data.TextToToken.TextToToken import TextToToken
import keras_nlp

class GptToken(TextToToken):
    def __init__(self):
        #No support for padding here
        self.bookmarked = False
        self.bookmark_tokens =[0,0]

        #Tokeniser
        self.gpt_token = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en")
        self.vocab_size = self.gpt_token#NEED TO DO THIS BUT WILL LOOK UP


    def init_with_input(self, input: list):
        pass

    def tokenise(self, input: list):
        return self.gpt_token(input)
    
    def detokenise(self, input: list) -> list:
        return self.gpt_token.detokenize(input)