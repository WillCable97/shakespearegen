import tensorflow as tf
import keras_nlp


# Example vocabulary
vocab = ["[PAD]", "[UNK]", "he", "hello", "low", "##llo", "w", "world"]

# Initialize WordpieceTokenizer with the vocabulary
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary = vocab)

# Get token for one element in the vocabulary
word = "hello"
tokens = tokenizer.tokenize([word])[0].numpy()

# Concatenate tokens to form the word
word_decoded = ""
for token_index in tokens:
    token = vocab[token_index]
    # Skip special tokens
    if not token.startswith("##"):
        word_decoded += token

print(f"Token for '{word}': {word_decoded}")



"""
import tensorflow as tf
import keras_nlp

# Example vocabulary
vocab = ["[PAD]", "[UNK]", "hello", "world", "how", "are", "you"]

# Initialize WordpieceTokenizer with the vocabulary
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary = vocab)#, sequence_length = self.sequence_len)
#text.WordpieceTokenizer(vocab)

# Get token for one element in the vocabulary
word = "hello"
token_index = tokenizer.tokenize([word])[0].numpy()[0]

# Get the token from the vocabulary using the index
token = tokenizer.detokenize([[token_index]]).numpy()[0].decode("utf-8")

print(f"Token for '{word}': {token}")
"""