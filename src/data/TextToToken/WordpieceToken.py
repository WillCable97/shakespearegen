from src.data.TextToToken.TextToToken import TextToToken

import keras
import keras_nlp
import tensorflow as tf

class WordpieceToken(TextToToken):
    def __init__(self, vocab_size: int, sequence_len: int):
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len

    def init_with_input(self, input: list):
        
        combined_input = input#[x[:20] for x in input][:20]
        tf_inputs = tf.data.Dataset.from_tensor_slices(combined_input)
        vocab_list = keras_nlp.tokenizers.compute_word_piece_vocabulary(tf_inputs, self.vocab_size)
        vocab_list.append("[START]")
        vocab_list.append("[END]")
        self.vocab_size = len(vocab_list)
        #print(f"vocab: \n {vocab_list[500:550]} \n ")
        self.wp_token = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary = vocab_list, sequence_length = self.sequence_len)

    def tokenise(self, input: list):
        return self.wp_token.tokenize(input)

    def detokenise(self, input: list) -> list:
        detoken = self.wp_token.detokenize(input)
        interm = detoken.numpy()
        interm = [f"{x}" for x in interm]
        return interm
    