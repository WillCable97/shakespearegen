import os 
import tensorflow as tf
import numpy as np

from src.data.TextToToken.WordpieceToken import WordpieceToken
from src.data.DataObjects.StandardTextDataObject import StandardTextDataObject
from src.data.DataLoaders import read_text_data

from src.models.RecurrentModels.RNN_model import RNN_model

from keras.losses import SparseCategoricalCrossentropy
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, OutputTextCallback
from src.models.TextGenerators.RecurrentNetworkGenerator import RecurrentNetworkGenerator


#Project details
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")
file_path = os.path.join(project_directory, "data", "processed", "linetext.txt")


model_name = "TEMPTHING"

sequence_length = 20
batch_size = 64
buffer_size = 10000
embedding_dimension = 256
dense_dimension = 512
epoch_count = 30

content_token = WordpieceToken(vocab_size=2000, sequence_len=sequence_length)


my_data_set = StandardTextDataObject(text_sequencer=content_token, data_loader=read_text_data
                                    , sequence_lenth=sequence_length, file_path = file_path)


#Hacky work around for padding isues with new tokeniser
flat_token_l = np.array(my_data_set.token_list).reshape(-1)
print(f"Flat Shape List: {flat_token_l.shape}")
flat_token_l = flat_token_l[flat_token_l != 0]
token_len = len(flat_token_l)
seq_count = int(token_len/sequence_length)
flat_token_l = flat_token_l[:seq_count * sequence_length]
flat_token_l = flat_token_l.reshape((seq_count, sequence_length))
print(f"Index Shape List: {flat_token_l.shape}")
my_data_set.token_list = flat_token_l.tolist()

my_data_set.create_tf_dataset()
my_data_set.create_label()
vocab_size_shake = my_data_set.text_sequencer.get_vocab_size()
print(f"Vocab: {vocab_size_shake}")

training_dataset = my_data_set.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)

lstm_inst = RNN_model(vocab_size=vocab_size_shake + 1, embedding_dim=embedding_dimension
                       , rnn_units=dense_dimension, batch_size=batch_size)

lstm_gen_inst = RNN_model(vocab_size=vocab_size_shake + 1, embedding_dim=embedding_dimension
                       , rnn_units=dense_dimension, batch_size=1) #This is because models need to be loaded into a model of batch size 1 tp produce output

#Compiling
loss_inst = SparseCategoricalCrossentropy(from_logits=True)
lstm_inst.compile("adam", loss=loss_inst)
lstm_gen_inst.build(tf.TensorShape([1, None])) 

#Callbacks
my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name,batch_size * 5)
test = RecurrentNetworkGenerator("the man went", lstm_gen_inst, content_token, 100)
output_callback = OutputTextCallback(test, project_directory, model_name)


print(test.generate_output())

#Fit model
lstm_inst.fit(training_dataset, epochs=epoch_count, callbacks=[my_csv_callback, my_checkpoint_callback, output_callback])