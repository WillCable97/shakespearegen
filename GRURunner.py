import os 
import tensorflow as tf
import numpy as np

from src.data.TextToToken.CustomCharacterToken import CustomCharacterToken
from src.data.DataObjects.StandardTextDataObject import E2EStandardTextObject
from src.data.DataLoaders import get_data_from_hgset

from src.models.RecurrentModels.GRU_model import GRU_model

from keras.losses import SparseCategoricalCrossentropy
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, OutputTextCallback
from src.models.TextGenerators.RecurrentNetworkGenerator import RecurrentNetworkGenerator


#Project details
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")
content_token = CustomCharacterToken(use_bookmark=False)

model_name = "GRU Model"

sequence_length = 100
batch_size = 64
buffer_size = 10000
embedding_dimension = 256
dense_dimension = 1024
epoch_count = 10

my_data_set = E2EStandardTextObject(text_sequencer=content_token, data_loader=get_data_from_hgset
                                    , sequence_lenth=sequence_length, set_name="tiny_shakespeare", sequence_len=sequence_length)


vocab_size_shake = my_data_set.text_sequencer.get_vocab_size()
print(f"Vocab: {vocab_size_shake}")

training_dataset = my_data_set.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)

lstm_inst = GRU_model(vocab_size=vocab_size_shake + 1, embedding_dim=embedding_dimension
                       , rnn_units=dense_dimension, batch_size=batch_size)

lstm_gen_inst = GRU_model(vocab_size=vocab_size_shake + 1, embedding_dim=embedding_dimension
                       , rnn_units=dense_dimension, batch_size=1) #This is because models need to be loaded into a model of batch size 1 tp produce output

#Compiling
loss_inst = SparseCategoricalCrossentropy(from_logits=True)
lstm_inst.compile("adam", loss=loss_inst)
lstm_gen_inst.build(tf.TensorShape([1, None])) 


#make a forward pass to init weights (I HATE THAT THIS IS NECESSARY !!!)
f_pass_output = lstm_gen_inst(np.array([my_data_set.token_list[0]]))


#Callbacks
my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name,batch_size * 5)
test = RecurrentNetworkGenerator("the man went", lstm_gen_inst, content_token, 100)
output_callback = OutputTextCallback(test, project_directory, model_name)

#Fit model
lstm_inst.fit(training_dataset, epochs=epoch_count, callbacks=[my_csv_callback, my_checkpoint_callback, output_callback])