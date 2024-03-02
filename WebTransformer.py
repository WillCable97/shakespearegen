import os 
import tensorflow as tf

from src.data.TextToToken.MyTfTokenizer import MyTfToken
from src.data.DataObjects.TextDataObject import TransformerTextDataObject
from src.data.DataLoaders import get_webscrape_data

from src.models.Transformer.Transformer import Transformer

from src.models.LossAndMetrics import masked_loss, masked_accuracy, CustomSchedule
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, TransformerOutputCallback
from src.models.TextGenerators.StandardTransformerGenerator import StandardTransformerGenerator


#Project details
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")
context_token = MyTfToken(use_bookmark=True)
content_token = MyTfToken(use_bookmark=True)

model_name = "WebTransformer"

sequence_length = 80
batch_size = 64
buffer_size = 10000
embedding_dimension = 128
dense_dimension = 256
num_heads = 2
num_att_layers = 2
dropout_rate = 0.1


my_data_set = TransformerTextDataObject(context_sequencer=context_token, content_sequencer=content_token
                                        , context_len=sequence_length, content_len=sequence_length
                                        ,data_loader=get_webscrape_data, data_path=path_to_data_folder)


vocab_size_shake = my_data_set.content_vocab
vocab_size_eng = my_data_set.context_vocab

print(f"Shakespeare Vocab: {vocab_size_shake} , English Vocab: {vocab_size_eng}")


training_dataset = my_data_set.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)

trans_inst = Transformer(vocab_size=vocab_size_shake, context_vocab_size=vocab_size_eng
                         ,embedding_dimension=embedding_dimension, context_length=sequence_length
                         ,content_length=sequence_length, num_heads=num_heads
                         , dense_dimension=dense_dimension, num_att_layers=num_att_layers)


learning_rate = CustomSchedule(embedding_dimension)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

trans_inst.compile(optimizer, loss=[masked_loss, None],metrics=["accuracy", masked_accuracy, None])
my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name,5)


tester= StandardTransformerGenerator(input_str="Hello this is my good friend he is a good person", source_model=trans_inst, output_len=sequence_length
                                     ,context_sequencer=my_data_set.context_sequencer, content_sequencer=my_data_set.content_sequencer)
output_callback = TransformerOutputCallback(tester, project_directory, model_name)

trans_inst.fit(training_dataset, epochs=10, callbacks=[my_csv_callback, my_checkpoint_callback, output_callback])
