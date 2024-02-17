import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


file_path = "./data/processed/linetext.txt"
sequence_length = 80
batch_size = 64
buffer_size = 10000
embedding_dimension = 256
dense_dimension = 256
num_heads = 2
num_att_layers = 2
dropout_rate = 0.1
tokenizer_used = Tokenizer()


def text_to_array(input_path: str) -> list:
    file = open(input_path, 'r')
    return file.readlines()


def text_array_from_file(input_path: str, len: int, tokenizer) -> np.array:
    array_text = text_to_array(input_path=input_path)
    tokenizer.fit_on_texts(array_text)
    tokenized_text = tokenizer.texts_to_sequences(array_text)
    padded_list = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, maxlen=len, padding="post")
    return padded_list



def create_offset_labels(input_tensor):
    return input_tensor[:-1], input_tensor[1:]


def create_final_dataset(input_path: str, sequence_length: int, batch_size: int, buffer_size: int, tokenizer):
    tokenized_text = text_array_from_file(input_path=input_path, len=sequence_length+1, tokenizer=tokenizer)
    tensor_slices = tf.data.Dataset.from_tensor_slices(tokenized_text)
    tensor_with_label = tensor_slices.map(create_offset_labels)
    final_set = tensor_with_label.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return final_set


data_set = create_final_dataset(input_path=file_path,sequence_length=sequence_length
                                ,batch_size=batch_size, buffer_size=buffer_size
                                ,tokenizer=tokenizer_used)


vocab_size = len(tokenizer_used.index_word)
vocab_size



def positional_encoding_matrix(sequence_length: int, embedding_dimension: int):
    #Create embedding vector
    effective_depth = embedding_dimension / 2
    depth_vector = np.repeat(np.arange(effective_depth), 2)
    frequency_vector = 1/(10000**((2* depth_vector)/embedding_dimension))

    #Token vector
    sequence_vector = np.arange(sequence_length)

    #Create matrix
    pos_encoding = sequence_vector.reshape([-1,1]) * frequency_vector.reshape([1,-1])
    pos_encoding[:,::2] = np.sin(pos_encoding[:,::2])
    pos_encoding[:,1::2] = np.cos(pos_encoding[:,1::2])
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, embedding_dimension: int, sequence_length: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dimension, mask_zero=True)
        self.positional_matrix = positional_encoding_matrix(sequence_length=sequence_length, embedding_dimension=embedding_dimension)

    def compute_mask(self, *args, **kwargs):
        return self.embedding_layer.compute_mask(*args, **kwargs)
    
    def call(self, x):
        #inner_seq_len = tf.shape(x)[1]
        x = self.embedding_layer(x)
        x = x + self.positional_matrix[tf.newaxis, :, :]
        return x


class AttentionBlockLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, embedding_dimension: int):
        super().__init__()
        self.multi_head_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dimension)
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        attention_output = self.multi_head_attn(query = x, value = x, key = x,
                                                use_causal_mask = True)
        x = self.add([x, attention_output])
        x = self.layer_norm(x)
        return x


class DenseComponent(tf.keras.layers.Layer):
    def __init__(self, embedding_dimension, dense_dimension, dropout_rate=0.1):
        super().__init__()
        #For som fkn reason sequential breaks the whole thing ??
        self.d1 = tf.keras.layers.Dense(dense_dimension, activation='relu')
        self.d1 = tf.keras.layers.Dense(embedding_dimension)
        self.d3 = tf.keras.layers.Dropout(dropout_rate)
        
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

        def call(self, x):
            x_1 = self.d1(x)
            x_1 = self.d2(x_1)
            x_1 = self.d3(x_1)
            x = self.add([x, x_1])
            x = self.layer_norm(x)
            return x


class DecoderComponent(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, embedding_dimension: int, dense_dimension: int):
        super(DecoderComponent, self).__init__()
        self.attention_block_layer = AttentionBlockLayer(num_heads=num_heads, embedding_dimension=embedding_dimension)
        self.dense_component = DenseComponent(embedding_dimension=embedding_dimension, dense_dimension=dense_dimension)

    def call(self, x):
        x = self.attention_block_layer(x)
        x = self.dense_component(x)
        return x



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, embedding_dimension: int, sequence_length: int
                 ,num_heads: int, dense_dimension: int, num_att_layers: int, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()


        self.embedding_dimension = embedding_dimension
        self.num_att_layers = num_att_layers

        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size, embedding_dimension=embedding_dimension
                                                        , sequence_length=sequence_length)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense_sequence = [DecoderComponent(num_heads=num_heads, embedding_dimension=embedding_dimension
                             , dense_dimension=dense_dimension) for i in range(num_att_layers)]
        
    def call(self, x):
        x = self.positional_embedding(x)
        x = self.dropout(x)
        
        for decoder in self.dense_sequence:
            x = decoder(x)

        return x


class SingleEmbeddedTransformer(tf.keras.Model):
    def __init__(self, vocab_size: int, embedding_dimension: int, sequence_length: int
                 ,num_heads: int, dense_dimension: int, num_att_layers: int, dropout_rate=0.1):
        super().__init__()  

        self.decoder_layer = DecoderLayer(vocab_size=vocab_size,embedding_dimension=embedding_dimension
                                          ,sequence_length=sequence_length,num_heads=num_heads
                                          ,dense_dimension=dense_dimension,num_att_layers=num_att_layers)
        self.final_dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.decoder_layer(x)
        logits = self.final_dense(x)
        
        try:
            del logits._keras_mask
        except AttributeError:
            pass
        
        return logits



def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss



single_emb_trans = SingleEmbeddedTransformer(vocab_size=vocab_size+1,embedding_dimension=embedding_dimension
                                             ,sequence_length=sequence_length
                                             ,num_heads=num_heads,dense_dimension=dense_dimension,num_att_layers=num_att_layers)



single_emb_trans.compile("adam", loss=[masked_loss, None],metrics=["accuracy", None])



single_emb_trans.fit(data_set,
                epochs=10)





def generate_text(input_text: str, max_length: int, generated_length: int) -> str:
    #Prepare the input vector
    tokenised_input = tokenizer_used.texts_to_sequences([input_text])[0]
    working_tokens = tokenised_input
    tokens_generated = []

    #Handle overflow strings
    initial_len = len(working_tokens)
    if len(working_tokens) > max_length: working_tokens = working_tokens[initial_len - max_length:]

    #Handle padding
    working_index = len(working_tokens)
    if working_index <= max_length: working_tokens = working_tokens + ([0] * (max_length - working_index))
    working_index-=1

    #Run generation loop
    for i in range(generated_length):
        np_arr = np.array(working_tokens)
        np_arr = np_arr[np.newaxis, :]

        prediction = single_emb_trans(np_arr)
        prediction_dimension = prediction[0][working_index]

        logits, indices = tf.math.top_k(prediction_dimension, k=10, sorted=True)
        indices = np.asarray(indices).astype("int32")
        pred_probs = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        pred_probs = np.asarray(pred_probs).astype("float32")
        chosen_token = np.random.choice(indices, p=pred_probs)

        tokens_generated.append(chosen_token)

        if working_index == max_length-1:
            working_tokens.append(chosen_token)
            working_tokens = working_tokens[1:]
        else:
            working_index += 1
            working_tokens[working_index] = chosen_token

    
    token_map = tokenizer_used.index_word   
    full_return = tokenised_input + tokens_generated
    full_return = [token_map[t] for t in full_return]
    return_str = ''
    for s in full_return: return_str += f"{s} "
    return return_str



input_test_string = "Romeo: "
print(generate_text(input_test_string, 80, 89))