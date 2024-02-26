#from src.data.TokenDataObj import TokenDataObj
#from keras.preprocessing.text import Tokenizer
#from src.models.Transformer.Transformer import Transformer
#from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback
#import tensorflow as tf
import os
#import pickle
from src.data.TextToToken.MyTfTokenizer import MyTfToken
from src.data.DataLoaders import get_webscrape_data
from src.data.DataObjects.TextDataObject import TransformerTextDataObject





#Project details
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")
context_token = MyTfToken()
content_token = MyTfToken()

model_name = "WebTransformer"


sequence_length = 80
batch_size = 64
buffer_size = 10000
embedding_dimension = 128
dense_dimension = 512
num_heads = 4
num_att_layers = 8
dropout_rate = 0.1

my_data_set = TransformerTextDataObject(context_sequencer=context_token, content_sequencer=content_token
                                        , context_len=sequence_length, content_len=sequence_length
                                        ,data_loader=get_webscrape_data, data_path=path_to_data_folder)




"""
#Get webscrape data lists
def get_webscrape_data(data_path: str):
    all_eng_text = []
    all_og_text =[]

    for dir_eg in os.listdir(data_path):
        path_to_play = os.path.join(data_path, dir_eg)
        
        with open(os.path.join(path_to_play, "english_lines.txt"), "rb") as fp:   # Unpickling
            play_eng_lines = pickle.load(fp)

        with open(os.path.join(path_to_play, "og_lines.txt"), "rb") as fp2:   # Unpickling
            play_og_lines = pickle.load(fp2)

        if len(play_eng_lines) != len(play_og_lines): print("PROBLEMS")

        all_eng_text += play_eng_lines
        all_og_text += play_og_lines
    return all_eng_text, all_og_text

all_eng_text, all_og_text = get_webscrape_data(path_to_data_folder)

print(all_eng_text[4321])
print("BREAK!!!")
print(all_og_text[4321])

"""














"""
all_eng_text = all_eng_text#[:8000]
all_og_text = all_og_text#[:8000]


#Process into formal data objects
def straight_loader(input):
    return input

context_data_obj = TokenDataObj(context_token)
content_data_obj = TokenDataObj(content_token)

context_data_obj.iniatiate_with_loader(straight_loader, input=all_eng_text)
content_data_obj.iniatiate_with_loader(straight_loader, input=all_og_text)

context_data_obj.final_dataset(sequence_length=sequence_length-1, batch_size=batch_size, buffer_size=buffer_size
                               ,padding_required=True, create_label=False, batching_required=False)

content_data_obj.final_dataset(sequence_length=sequence_length, batch_size=batch_size, buffer_size=buffer_size
                               ,padding_required=True, create_label=True, batching_required=False)


vocab_size_shake = content_data_obj.vocab_size
vocab_size_eng = context_data_obj.vocab_size

print(f"VOCABS: {vocab_size_shake}, {vocab_size_eng}")


context_vector = context_data_obj.final_dataset_obj
content_vector = content_data_obj.final_dataset_obj
prelim_total = tf.data.Dataset.zip((context_vector, content_vector))

def reorder_stuff(context_ten, content_ten):
   a = context_ten
   (b, c) = content_ten
   return (a,b) , c
   
prelim_total2 = prelim_total.map(reorder_stuff)
total_vecor = prelim_total2.shuffle(buffer_size).batch(batch_size, drop_remainder=True)


#context_vector = context_data_obj.final_dataset_obj
#content_vector = content_data_obj.final_dataset_obj
#label_vector = content_data_obj.final_dataset_obj
#content_vector = content_vector.map(lambda x: x[:-1])
#label_vector = label_vector.map(lambda x: x[1:])

#inputs_vector = tf.data.Dataset.zip((context_vector, content_vector))
#total_vecor = tf.data.Dataset.zip((inputs_vector, label_vector))
#total_vecor = total_vecor.shuffle(buffer_size).batch(batch_size, drop_remainder=True)












print(total_vecor)


trans_inst = Transformer(vocab_size=vocab_size_shake+1, context_vocab_size=vocab_size_eng+1,
                         embedding_dimension=embedding_dimension, sequence_length=sequence_length,
                         num_heads=num_heads, dense_dimension=dense_dimension, num_att_layers=num_att_layers)


#Training Metrics
def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(embedding_dimension)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

trans_inst.compile(optimizer, loss=[masked_loss, None],metrics=["accuracy", None]) #Dont want ADAM optimizer!!!

my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name)


import numpy as np
class TextGenThing:
    def __init__(self, sequence_len: int, start_str: str, tokenizers: list[Tokenizer], input_model: tf.keras.models.Model):
        self.sequence_len = sequence_len
        self.start_str = start_str
        self.constext_token, self.content_token = tokenizers
        self.context_input = self.create_context_input()
        self.initial_content = np.array([[1]])

        print(f"{self.context_input.shape} : {self.initial_content.shape}")
        self.A = input_model([self.context_input, self.initial_content])
        print(self.A)
        
    def create_context_input(self) -> np.array:
        tokenized_text = self.constext_token.texts_to_sequences([self.start_str])
        padded_input = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, maxlen=self.sequence_len, padding="post")
        return np.array(padded_input)
   
      

Test = TextGenThing(sequence_len=sequence_length, start_str="I am", tokenizers=[context_token, content_token], input_model = trans_inst)



"""





















#####################THIS WAS REALLY ARCHIVED











































#print(context_token.texts_to_sequences(['']))




"""a = None

for A in total_vecor.take(1):
   a0,b= A
   a = a0
   #c=trans_inst(a0)
   #trans_inst.summary()
   break

print(a)
"""
#trans_inst.fit(total_vecor, epochs=3, callbacks=[my_csv_callback, my_checkpoint_callback])



#Input Tokens 
#(Context Tokens)
#Output Text !! 

#Sequence length (input)
#Beginning sentence
#Tokenizer (s)

#Function of output stays fixed


"""

class TextGenThing:
   def __init__(self, sequence_len: int, start_str: str, tokenizers: list[Tokenizer]):
      pass

Test = TextGenThing(sequence_len=sequence_length, start_str="I am", tokenizers=[context_token, content_token])





"""




"""
import keras
class TextGenerator(keras.callbacks.Callback):
    def __init__(
        self, max_tokens, start_sentence, tokenizer, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.print_every = print_every
        self.k = top_k
        self.start_tokens = tokenizer.texts_to_sequences([start_sentence.split()])[0]

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)
    
    def generate_text_from_texts(self, texts):
        tokens = tokenizer.texts_to_sequences([texts.split()])[0]
        return generate_text_from_tokens(tokens)
    
    def generate_text_from_tokens(self, start_tokens):
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.max_tokens - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.max_tokens]
                sample_index = self.max_tokens - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = self.tokenizer.sequences_to_texts([start_tokens + tokens_generated])[0]
        return txt

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        txt = self.generate_text_from_tokens(self.start_tokens)
        print(f"generated text:\n{txt}\n")
text_generator = TextGenerator(Config.maxlen, "I am", tokenizer)
"""










#trans_inst.fit(total_vecor, epochs=3, callbacks=[my_csv_callback, my_checkpoint_callback])
