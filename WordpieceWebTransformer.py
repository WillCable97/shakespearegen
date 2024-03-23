import os 
import tensorflow as tf

from src.data.TextToToken.WordpieceToken import WordpieceToken
from src.data.DataObjects.TransformerTextDataObject import TransformerTextDataObject
from src.data.DataLoadersDir.WebscrapeData import base_webscrape_with_ends, training_set, val_set, overlap_with_ends
#from src.data.DataLoaders import get_webscrape_data, get_webscrape_data_withends,

from src.models.Transformer.Transformer import Transformer

from src.models.LossAndMetrics import masked_loss, masked_accuracy, CustomSchedule
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, OutputTextCallback
from src.models.TextGenerators.StandardTransformerGenerator import StandardTransformerGenerator


#Project details
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")

model_name = "WO_P_T_M1.0"

sequence_length = 150
batch_size = 64
buffer_size = 10000
embedding_dimension = 128
dense_dimension = 256
num_heads = 4
num_att_layers = 2
dropout_rate = 0.1
epoch_count = 40

context_token = WordpieceToken(vocab_size=5000, sequence_len=sequence_length)
content_token = WordpieceToken(vocab_size=5000, sequence_len=sequence_length)

my_data_set = TransformerTextDataObject(context_sequencer=context_token, content_sequencer=content_token
                                        , context_len=sequence_length, content_len=sequence_length
                                        , data_loader = training_set,input_datafetcher=overlap_with_ends, data_path=path_to_data_folder
                                        , training_proportion= 0.9)

my_val_set = TransformerTextDataObject(context_sequencer=context_token, content_sequencer=content_token
                                        , context_len=sequence_length, content_len=sequence_length
                                        , data_loader = val_set, input_datafetcher=overlap_with_ends, data_path=path_to_data_folder
                                        , val_proportion= 0.1)

vocab_size_shake = my_data_set.content_vocab
vocab_size_eng = my_data_set.context_vocab
print(f"Shakespeare Vocab: {vocab_size_shake} , English Vocab: {vocab_size_eng}")

training_dataset = my_data_set.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)
val_dataset = my_val_set.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)

#print(my_data_set.token_context[:100])


trans_inst = Transformer(vocab_size=vocab_size_shake, context_vocab_size=vocab_size_eng
                         ,embedding_dimension=embedding_dimension, context_length=sequence_length
                         ,content_length=sequence_length, num_heads=num_heads
                         , dense_dimension=dense_dimension, num_att_layers=num_att_layers)


learning_rate = CustomSchedule(embedding_dimension)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


trans_inst.compile(optimizer, loss=[masked_loss],metrics=[masked_accuracy])
my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name,5)

tester= StandardTransformerGenerator(input_str="hello this is my brother, he is a good person", source_model=trans_inst, output_len=sequence_length
                                     ,context_sequencer=my_data_set.context_sequencer, content_sequencer=my_data_set.content_sequencer)
output_callback = OutputTextCallback(tester, project_directory, model_name)

trans_inst.fit(training_dataset, validation_data=val_dataset, epochs=epoch_count
               , callbacks=[my_csv_callback, my_checkpoint_callback, output_callback])
