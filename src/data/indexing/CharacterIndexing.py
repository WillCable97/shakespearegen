from IndexBase import TextIndexer

#Make this into a datacalss as well !!!????
class CharacterIndexing(TextIndexer): #Will need to fix for cases where string is apart of a n dim list
    def __init__(self):
        self.data_init = False

    def create_mapping(self, input_text: str) -> None:
        full_text = ''.join(input_text)
        ordered_text = sorted(set((full_text)))
        text_to_int = {c: i for i,c in enumerate(ordered_text)}
        int_to_text = {text_to_int[c] : c for c in text_to_int}
        
        self.mapped_text = [[text_to_int[c] for c in sentence] for sentence in input_text]

        #self.mapped_text = [[text_to_int[c] for c in str_f] for sentence in input_text for str_f in sentence]
        self.text_to_int = text_to_int
        self.int_to_text = int_to_text
        self.data_init = True
        return 
        
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