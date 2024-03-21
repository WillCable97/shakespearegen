import os 
import tensorflow as tf

from src.data.TextToToken.WordpieceToken import WordpieceToken
from src.models.Transformer.Transformer import Transformer
from src.models.TextGenerators.StandardTransformerGenerator import StandardTransformerGenerator
from src.data.DataObjects.TransformerTextDataObject import TransformerTextDataObject

from src.data.DataLoaders import get_webscrape_data, get_webscrape_data_withends

project_directory = os.path.abspath("./")
path_to_weight = os.path.join(project_directory,"models", "OnlineArchi", "checkpoint_tracker", "ckpt_20.weights.h5")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")

sequence_length = 30
embedding_dimension = 128
dense_dimension = 512
num_heads = 8
num_att_layers = 4

context_token = WordpieceToken(vocab_size=5000, sequence_len=sequence_length)
content_token = WordpieceToken(vocab_size=5000, sequence_len=sequence_length)

trans_inst = Transformer(vocab_size=4792, context_vocab_size=4901
                         ,embedding_dimension=embedding_dimension, context_length=sequence_length
                         ,content_length=sequence_length, num_heads=num_heads
                         , dense_dimension=dense_dimension, num_att_layers=num_att_layers)

my_data_set = TransformerTextDataObject(context_sequencer=context_token, content_sequencer=content_token
                                        , context_len=sequence_length, content_len=sequence_length
                                        ,data_loader=get_webscrape_data_withends, data_path=path_to_data_folder)



tester= StandardTransformerGenerator(input_str="hello this is my brother, he is a good person", source_model=trans_inst, output_len=sequence_length
                                     ,context_sequencer=my_data_set.context_sequencer, content_sequencer=my_data_set.content_sequencer)

#source_model
print(tester.generate_output())

tester.source_model.load_weights(path_to_weight)

test_strs = ["hello this is my brother, he is a good person"
             ,"this morning the bird went to her nest and laid an egg"
             ,"Last year I went on holiday to another country for one week. It was very relaxing and I would like to go back"
             ,"excuse me, do you know the time?"
             ,"I'm hungry, I think I'll go get some lunch"
             ,"I think it will rain next week"]



for test_str in test_strs:
    tester.input_str = test_str
    tester.context_vector=tester.create_context_vector()
    print(tester.generate_output())






