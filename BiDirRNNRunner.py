import os 
import tensorflow as tf
import numpy as np

from src.data.TextToToken.CustomCharacterToken import CustomCharacterToken
from src.data.TextToToken.MyTfTokenizer import MyTfToken
from src.data.DataObjects.StandardTextDataObject import E2EStandardTextObject, StandardTextDataObject
from src.data.DataLoaders import get_data_from_hgset, identity_loader, read_text_with_sequences

from src.models.RecurrentModels.RNN_model import RNN_model
from src.models.RecurrentModels.LSTM_model import LSTM_model
from src.models.RecurrentModels.LSTM_model_Birdir import LSTM_model_Birdir
#from src.models.RecurrentModels.RNN_model import RNN_model

from keras.losses import SparseCategoricalCrossentropy
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, OutputTextCallback
from src.models.TextGenerators.RecurrentNetworkGenerator import RecurrentNetworkGenerator


#Project details
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")
#content_token = CustomCharacterToken(use_bookmark=False)
#val_token = CustomCharacterToken(use_bookmark=False)
content_token = MyTfToken(use_bookmark=False)
val_token = MyTfToken(use_bookmark=False)



model_name = "BIDIRTEST"

sequence_length = 30
batch_size = 64
buffer_size = 10000
embedding_dimension = 256
dense_dimension = 512
kernel_regularizer = 0#0.01
dropout = 0.2
epoch_count = 30
validation_prop=0.05


text_path = os.path.join(project_directory, "data/processed/linetext.txt")

my_data_set = E2EStandardTextObject(text_sequencer=content_token, data_loader=read_text_with_sequences
                                    , sequence_lenth=sequence_length, validation_prop=0.2
                                    , file_path=text_path, sequence_len=sequence_length, len_type = "word")

#my_data_set = E2EStandardTextObject(text_sequencer=content_token, data_loader=get_data_from_hgset
#                                    , sequence_lenth=sequence_length, validation_prop=0.05
#                                    , set_name="tiny_shakespeare", sequence_len=sequence_length)

raw_val_set = my_data_set.val_raw_set

val_data_set = E2EStandardTextObject(text_sequencer=val_token, data_loader=identity_loader
                                     , sequence_lenth=sequence_length, input = raw_val_set) #IF I PUT THIS FIRST THE VOCAB SIZE IS WRONG. WHY??????

val_data_set.text_sequencer = my_data_set.text_sequencer

training_dataset = my_data_set.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)
validation_dataset = val_data_set.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)

vocab_size_shake = my_data_set.text_sequencer.get_vocab_size()
print(f"Vocab: {vocab_size_shake}")


lstm_inst = LSTM_model_Birdir(vocab_size=vocab_size_shake + 1, embedding_dim=embedding_dimension
                       , rnn_units=dense_dimension, batch_size=batch_size
                       , regularizer=kernel_regularizer,dropout_rate=dropout)

lstm_gen_inst = LSTM_model_Birdir(vocab_size=vocab_size_shake + 1, embedding_dim=embedding_dimension
                       , rnn_units=dense_dimension, batch_size=1
                       , regularizer=kernel_regularizer,dropout_rate=dropout) #This is because models need to be loaded into a model of batch size 1 tp produce output

#Compiling
loss_inst = SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(0.01)

lstm_inst.compile("adam", loss=loss_inst, metrics=["accuracy"])

lstm_gen_inst.build(tf.TensorShape([1, None])) 

#Callbacks
my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name,batch_size * 5)
test = RecurrentNetworkGenerator("the man went", lstm_gen_inst, content_token, 100)
output_callback = OutputTextCallback(test, project_directory, model_name)

#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)

#Fit model
lstm_inst.fit(training_dataset, epochs=epoch_count, callbacks=[my_csv_callback, my_checkpoint_callback, output_callback]
              ,validation_data=validation_dataset)