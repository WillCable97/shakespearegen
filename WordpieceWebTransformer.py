import os 
import tensorflow as tf

from src.data.TextToToken.WordpieceToken import WordpieceToken
from src.data.DataObjects.TransformerTextDataObject import TransformerTextDataObject
from src.data.DataLoaders import get_webscrape_data, get_webscrape_data_withends

from src.models.Transformer.Transformer import Transformer

from src.models.LossAndMetrics import masked_loss, masked_accuracy, CustomSchedule
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, OutputTextCallback
from src.models.TextGenerators.StandardTransformerGenerator import StandardTransformerGenerator


#Project details
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")

model_name = "Worpiece_Trans_V2"

sequence_length = 30
batch_size = 64
buffer_size = 10000
embedding_dimension = 128
dense_dimension = 512
num_heads = 4
num_att_layers = 4
dropout_rate = 0.1
epoch_count = 20

context_token = WordpieceToken(vocab_size=5000, sequence_len=sequence_length)
content_token = WordpieceToken(vocab_size=5000, sequence_len=sequence_length)

#all_eng_text, all_og_text = get_webscrape_data(data_path=path_to_data_folder)
#context_token.init_with_input(all_eng_text)

my_data_set = TransformerTextDataObject(context_sequencer=context_token, content_sequencer=content_token
                                        , context_len=sequence_length, content_len=sequence_length
                                        ,data_loader=get_webscrape_data_withends, data_path=path_to_data_folder)

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

trans_inst.compile(optimizer, loss=[masked_loss],metrics=["accuracy", masked_accuracy])
my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name,5)

a, b = get_webscrape_data(data_path=path_to_data_folder)

tester= StandardTransformerGenerator(input_str="hello this is my brother, he is a good person", source_model=trans_inst, output_len=sequence_length
                                     ,context_sequencer=my_data_set.context_sequencer, content_sequencer=my_data_set.content_sequencer)
output_callback = OutputTextCallback(tester, project_directory, model_name)


tester2= StandardTransformerGenerator(input_str="this morning the bird went to her nest and laid an egg", source_model=trans_inst, output_len=sequence_length
                                     ,context_sequencer=my_data_set.context_sequencer, content_sequencer=my_data_set.content_sequencer)
output_callback2 = OutputTextCallback(tester2, project_directory, model_name)


tester3= StandardTransformerGenerator(input_str="Last year I went on holiday to another country for one week. It was very relaxing and I would like to go back", source_model=trans_inst, output_len=sequence_length
                                     ,context_sequencer=my_data_set.context_sequencer, content_sequencer=my_data_set.content_sequencer)
output_callback3 = OutputTextCallback(tester3, project_directory, model_name)

trans_inst.fit(training_dataset, epochs=epoch_count, callbacks=[my_csv_callback, my_checkpoint_callback, output_callback3, output_callback2, output_callback])