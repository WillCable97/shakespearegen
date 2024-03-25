from src.data.TextToToken.TextToToken import TextToToken
from transformers import GPT2Tokenizer

class GptToken(TextToToken):
    def __init__(self):
        #No support for padding here
        self.bookmarked = False
        self.bookmark_tokens =[0,0]

        #Tokeniser
        self.gpt_token = GPT2Tokenizer.from_pretrained("gpt2")
        print(len(self.gpt_token.get_vocab()))
        self.gpt_token.pad_token_id= 0
        #self.gpt_token.add_special_tokens({'pad_token': '[PAD]'})
        self.eos_token_id = self.gpt_token.eos_token_id
        print(len(self.gpt_token.get_vocab()))
        self.vocab_size = len(self.gpt_token.get_vocab())

    def init_with_input(self, input: list):
        pass

    def tokenise(self, input: list):
        tok_output = self.gpt_token(input, padding="max_length", truncation=True, max_length=256)
        return list(tok_output.data.values())[0] #Dont need to return the mask
    
    def detokenise(self, input: list) -> list:
        ret_list = [self.gpt_token.decode(x) for x in input]
        return ret_list#self.gpt_token.detokenize(input)
    

