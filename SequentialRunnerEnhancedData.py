from src.data.DataLoadersDir.EnhancedText import complete_single_emb_retriever
from src.data.TextToToken.CustomCharacterToken import CustomCharacterToken
from src.data.DataObjects.StandardTextDataObject import E2EStandardTextObject

from src.models.RecurrentModels.RNN_model import RNN_model
from src.models.RecurrentModels.GRU_model import GRU_model
from src.models.RecurrentModels.LSTM_model import LSTM_model
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, OutputTextCallback
from src.models.TextGenerators.RecurrentNetworkGenerator import RecurrentNetworkGenerator

import os
import tensorflow as tf
import keras

#Meta Info
model_name = "W_P_RNN100_S1.0"

#Data hyperparameters
data_soure = "HuggingFace"
data_sequencing_len = 100

#Pre processing hyperparameters
token_seqence_length = data_sequencing_len
batch_size = 64
buffer_size = 10000

#Model hyperparameters
embedding_dimension = 128
dense_dimension = 512
#kernel_regularizer = 0#0.01
#dropout = 0.2
epoch_count = 40


#File path values
root_dir = os.path.abspath("./")
processed_data = os.path.join(root_dir, "data", "processed")

#************************DATA************************

#Tokens (Text to sequence)
char_token = CustomCharacterToken()

#Data objects
base_data_object = E2EStandardTextObject(text_sequencer=char_token, data_loader=complete_single_emb_retriever
                                         , sequence_lenth=token_seqence_length
                                         , base_path=processed_data, data_source=data_soure, data_sequencing_len=data_sequencing_len
                                         , set_suffix = "base")


validation_data_object = E2EStandardTextObject(text_sequencer=base_data_object.text_sequencer, data_loader=complete_single_emb_retriever
                                               , sequence_lenth=token_seqence_length, init_tokens=False
                                               , base_path=processed_data, data_source=data_soure, data_sequencing_len=data_sequencing_len
                                               , set_suffix = "val")

training_data_object = E2EStandardTextObject(text_sequencer=base_data_object.text_sequencer, data_loader=complete_single_emb_retriever
                                             , sequence_lenth=token_seqence_length, init_tokens=False
                                             , base_path=processed_data, data_source=data_soure, data_sequencing_len=data_sequencing_len)


vocab_size = base_data_object.vocab_size
print(f"Vocab: {vocab_size}")


training_dataset = training_data_object.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)
validation_dataset = validation_data_object.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)


#************************MODEL COMPILATION************************

#Model definition
sequential_inst = RNN_model(vocab_size=vocab_size+1, embedding_dim=embedding_dimension, rnn_units=dense_dimension
                            ,batch_size=batch_size)

sequential_gen_inst = RNN_model(vocab_size=vocab_size+1, embedding_dim=embedding_dimension, rnn_units=dense_dimension
                                ,batch_size=1)

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
sequential_inst.compile("adam", loss=loss_obj, metrics="accuracy")
sequential_gen_inst.build(tf.TensorShape([1, None])) 


#************************MODEL CALLBACKS************************

my_csv_callback = csv_callback(root_dir, model_name)
my_checkpoint_callback = checkpoint_callback(root_dir, model_name,5)


tester= RecurrentNetworkGenerator(input_str="the man went", source_model=sequential_gen_inst
                                  , sequencer=training_data_object.text_sequencer, output_len=150)
output_callback = OutputTextCallback(tester, root_dir, model_name)

sequential_inst.fit(training_dataset, epochs=epoch_count, validation_data=validation_dataset
                    , callbacks=[my_csv_callback, my_checkpoint_callback, output_callback])
