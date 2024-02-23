from src.data.TokenDataObj import TokenDataObj
from keras.preprocessing.text import Tokenizer
from src.models.Transformer.Transformer import Transformer
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback
import tensorflow as tf
import os
import pickle

#Project details
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")
context_token = Tokenizer()
content_token = Tokenizer()

model_name = "WebTransformer"


sequence_length = 80
batch_size = 64
buffer_size = 10000
embedding_dimension = 256
dense_dimension = 256
num_heads = 2
num_att_layers = 2
dropout_rate = 0.1

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


all_eng_text = all_eng_text#[:10000]
all_og_text = all_og_text#[:10000]


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
                               ,padding_required=True, create_label=False, batching_required=False)

vocab_size_shake = content_data_obj.vocab_size
vocab_size_eng = context_data_obj.vocab_size

context_vector = context_data_obj.final_dataset_obj
content_vector = content_data_obj.final_dataset_obj
label_vector = content_data_obj.final_dataset_obj
content_vector = content_vector.map(lambda x: x[:-1])
label_vector = label_vector.map(lambda x: x[1:])

inputs_vector = tf.data.Dataset.zip((context_vector, content_vector))
total_vecor = tf.data.Dataset.zip((inputs_vector, label_vector))
total_vecor = total_vecor.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

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


trans_inst.fit(total_vecor, epochs=3, callbacks=[my_csv_callback, my_checkpoint_callback])

trans_inst.save_weights("./modelsave/modelsave")


















"""



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


trans_inst.compile("adam", loss=[masked_loss, None],metrics=["accuracy", None])
trans_inst.fit(final_d_set, epochs=3)
"""


"""

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



my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name)


single_emb_trans.compile("adam", loss=[masked_loss, None],metrics=["accuracy", None])
single_emb_trans.fit(test_text.final_dataset_obj, epochs=3, callbacks=[my_csv_callback, my_checkpoint_callback])
"""

