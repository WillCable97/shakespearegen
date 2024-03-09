import os 
import tensorflow as tf

from src.data.TextToToken.MyTfTokenizer import MyTfToken
from src.data.DataObjects.TransformerTextDataObject import TransformerTextDataObject
from src.data.DataLoaders import get_webscrape_data

from src.models.Seq2Seq.Seq2Seq import Seq2SeqModel

from src.models.LossAndMetrics import masked_loss, masked_accuracy, CustomSchedule
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, OutputTextCallback
from src.models.TextGenerators.StandardTransformerGenerator import StandardTransformerGenerator


#Project details
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")
context_token = MyTfToken(use_bookmark=True)
content_token = MyTfToken(use_bookmark=True)

model_name = "Seq2SeqModel"

sequence_length = 62
batch_size = 64
buffer_size = 10000
embedding_dimension = 128
dense_dimension = 256
num_heads = 2
num_layers = 1
dropout_rate = 0.1


my_data_set = TransformerTextDataObject(context_sequencer=context_token, content_sequencer=content_token
                                        , context_len=sequence_length, content_len=sequence_length
                                        ,data_loader=get_webscrape_data, data_path=path_to_data_folder)

vocab_size_shake = my_data_set.content_vocab
vocab_size_eng = my_data_set.context_vocab

print(f"Shakespeare Vocab: {vocab_size_shake} , English Vocab: {vocab_size_eng}")

training_dataset = my_data_set.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)

trans_inst = Seq2SeqModel(content_vocab=vocab_size_shake, context_vocab=vocab_size_eng
                         ,embedding_dimension=embedding_dimension, context_length=sequence_length
                         ,content_length=sequence_length, dense_dimension=dense_dimension, num_layers=num_layers)

trans_inst.compile("adam", loss=[masked_loss, None],metrics=["accuracy", masked_accuracy, None])
my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name,5)


tester= StandardTransformerGenerator(input_str="hello this is my brother", source_model=trans_inst, output_len=sequence_length
                                     ,context_sequencer=my_data_set.context_sequencer, content_sequencer=my_data_set.content_sequencer)
output_callback = OutputTextCallback(tester, project_directory, model_name)

trans_inst.fit(training_dataset, epochs=10, callbacks=[my_csv_callback, my_checkpoint_callback, output_callback])