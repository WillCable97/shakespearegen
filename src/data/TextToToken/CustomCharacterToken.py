from src.data.TextToToken.TextToToken import TextToToken

class CustomCharacterToken(TextToToken):
    """
        Tokeniser for charachter level data
    """
    def __init__(self, use_bookmark: bool = False):
        self.bookmarked = use_bookmark

    def init_with_input(self, input: list):
        full_text = ' '.join(input)
        ordered_text = sorted(set(full_text))
        word_index = {c: i+1 for i,c in enumerate(ordered_text)}
        index_word = {word_index[c]: c for c in word_index}
        self.vocab_size = len(word_index)
        self.word_index = word_index
        self.index_word = index_word
        self.bookmark_tokens = [self.vocab_size + 1, self.vocab_size + 2] 
        if self.bookmarked: self.vocab_size += 2 #this is to account for the extra 2 tokens on either side

    def tokenise(self, input: list):
        return [[self.word_index[char_found] for char_found in sentence] for sentence in input]
    
    def detokenise(self, input: list) -> list:
        split_list = [['' if char_found in self.bookmark_tokens else self.index_word[char_found] 
                       for char_found in sentence] for sentence in input]
        return [''.join(split_sent) for split_sent in split_list]


