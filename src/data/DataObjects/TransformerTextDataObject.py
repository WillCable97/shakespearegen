import typing
import tensorflow as tf

from src.data.DataObjects.helper_funcs import reorder_transformer_dataset, create_offset_labels
from src.data.TextToToken.TextToToken import TextToToken

class TransformerTextDataObject:
    def __init__(self, context_sequencer: TextToToken, content_sequencer: TextToToken
                 , context_len: int, content_len: int, data_loader: typing.Callable[[typing.Any], list],**kwargs):
        #Raw data and sequencing
        self.context_sequencer = context_sequencer
        self.content_sequencer = content_sequencer
        self.context_len = context_len
        self.content_len = content_len + 1 #One will be lost when the map is applied to create labels
        self.raw_context, self.raw_content = data_loader(**kwargs)
        self.context_sequencer.init_with_input(self.raw_context)
        self.content_sequencer.init_with_input(self.raw_content)
        self.token_context = self.context_sequencer.tokenise(self.raw_context)
        self.token_content = self.content_sequencer.tokenise(self.raw_content)

        #Vocabs
        self.context_vocab = self.context_sequencer.get_vocab_size()
        self.content_vocab = self.content_sequencer.get_vocab_size()

        #Create data by default
        self.pad_sequences()
        self.create_tf_dataset()
        self.create_label()
        
    def pad_sequences(self) -> None:
        token_vecs = [self.token_context, self.token_content]
        lens = [self.context_len, self.content_len]
        bookmarks = [self.context_sequencer.bookmark_status(), self.content_sequencer.bookmark_status()]
        token_vals = [self.context_sequencer.get_bookmark_tokens(), self.content_sequencer.get_bookmark_tokens()]

        for i, vec in enumerate(token_vecs):
            if bookmarks[i]: vec = [seq + [token_vals[i][1]] for seq in vec]
            vec = tf.keras.preprocessing.sequence.pad_sequences(vec, maxlen=lens[i]
                                                              ,padding="post")
            if bookmarks[i]:
                for j, _ in enumerate(vec): vec[j][0] = token_vals[i][0]
            token_vecs[i] = vec
        
        self.token_context = token_vecs[0]
        self.token_content = token_vecs[1]

    def unpad_sequance(self) -> None:
        pass

    def create_tf_dataset(self):
        self.context_tf_dataset = tf.data.Dataset.from_tensor_slices(self.token_context)
        self.content_tf_dataset = tf.data.Dataset.from_tensor_slices(self.token_content)

    def create_label(self):
        self.content_tf_dataset = self.content_tf_dataset.map(create_offset_labels)

    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        preliminary_tensor = tf.data.Dataset.zip((self.context_tf_dataset, self.content_tf_dataset))
        preliminary_tensor = preliminary_tensor.map(reorder_transformer_dataset)
        self.final_tf_dataset = preliminary_tensor.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return self.final_tf_dataset