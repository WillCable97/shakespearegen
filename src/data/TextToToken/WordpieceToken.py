from src.data.TextToToken.TextToToken import TextToToken

import keras
import keras_nlp
import tensorflow as tf

class WordpieceToken(TextToToken):
    def __init__(self, vocab_size: int, sequence_len: int) :
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.bookmarked = False

    def init_with_input(self, input: list):
        combined_input = input#[x[:20] for x in input][:20]
        tf_inputs = tf.data.Dataset.from_tensor_slices(combined_input)
        reserved_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "*", "[START]", "[END]"]

        vocab_list = keras_nlp.tokenizers.compute_word_piece_vocabulary(tf_inputs, self.vocab_size, reserved_tokens=reserved_tokens)
        self.vocab_size = len(vocab_list)
        self.wp_token = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary = vocab_list, sequence_length = self.sequence_len)
        
        self.bookmark_tokens = [5,5] #This is the index of the reserved tokens (be careful here)

    def tokenise(self, input: list):
        return self.wp_token.tokenize(input)

    def detokenise(self, input: list) -> list:
        detoken = self.wp_token.detokenize(input)
        interm = detoken.numpy()
        interm = [x.decode('utf-8') for x in interm]
        return interm
    