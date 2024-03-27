import os 
import tensorflow as tf

from src.data.TextToToken.WordpieceToken import WordpieceToken
from src.models.Transformer.Transformer import Transformer
from src.models.TextGenerators.StandardTransformerGenerator import StandardTransformerGenerator
from src.data.DataObjects.TransformerTextDataObject import TransformerTextDataObject

from src.data.DataLoaders import get_webscrape_data, get_webscrape_data_withends



#Data hyperparameters
data_soure = "Webscrape"
data_sequencing_len = 1

#Pre processing hyperparameters
token_seqence_length = 75
batch_size = 64
buffer_size = 10000

#Model hyperparameters
embedding_dimension = 64
dense_dimension = 64
num_heads = 1
num_att_layers = 1
dropout_rate = 0.1
epoch_count = 40



#Model definition
trans_inst = Transformer(vocab_size=vocab_size_shake, context_vocab_size=vocab_size_eng
                         , embedding_dimension=embedding_dimension,  dense_dimension=dense_dimension
                         , context_length=token_seqence_length,content_length=token_seqence_length
                         , num_att_layers=num_att_layers, num_heads=num_heads)


















"""
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")

sequence_length = 30
embedding_dimension = 128
dense_dimension = 512
num_heads = 8
num_att_layers = 4

context_token = WordpieceToken(vocab_size=5000, sequence_len=sequence_length)
content_token = WordpieceToken(vocab_size=5000, sequence_len=sequence_length)

trans_inst = Transformer(vocab_size=4866, context_vocab_size=4821
                         ,embedding_dimension=embedding_dimension, context_length=sequence_length
                         ,content_length=sequence_length, num_heads=num_heads
                         , dense_dimension=dense_dimension, num_att_layers=num_att_layers)

my_data_set = TransformerTextDataObject(context_sequencer=context_token, content_sequencer=content_token
                                        , context_len=sequence_length, content_len=sequence_length
                                        ,data_loader=get_webscrape_data, data_path=path_to_data_folder)



tester= StandardTransformerGenerator(input_str="hello this is my brother, he is a good person", source_model=trans_inst, output_len=sequence_length
                                     ,context_sequencer=my_data_set.context_sequencer, content_sequencer=my_data_set.content_sequencer)


print(tester.generate_output())

test_strs = ["hello this is my brother, he is a good person"
             ,"this morning the bird went to her nest and laid an egg"
             ,"Last year I went on holiday to another country for one week. It was very relaxing and I would like to go back"
             ,"excuse me, do you know the time?"
             ,"I'm hungry, I think I'll go get some lunch"
             ,"I think it will rain next week"]

lines = ["LINES TO CREATE: \n"] + ["- " + x + "\n" for x in test_strs] + ["\n\n"]

print("ASDFAS")

for i in range(3):
    print(f"EPOCH {i}")
    path_to_weight = os.path.join(project_directory,"models", "W_P_T_M1.0", "checkpoint_tracker", f"ckpt_{i+1}.weights.h5")
    tester.source_model.load_weights(path_to_weight)
    lines.append(f"EPOCH : {i+1}\n")

    for test_str in test_strs:
        tester.input_str = test_str
        tester.context_vector=tester.create_context_vector()
        lines.append(tester.generate_output() + "\n")
    
    lines.append("\n\n")


path_to_lines = os.path.join(project_directory,"models", "W_P_T_M1.0", "generated_lines.txt")


print(lines)

with open(path_to_lines, "w+") as file:
    file.writelines(lines)

"""