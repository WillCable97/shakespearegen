from src.data.DataLoadersDir.EnhancedText import complete_transformer_retriever
from src.data.TextToToken.WordpieceToken import WordpieceToken
#from src.data.TextToToken.CustomCharacterToken import CustomCharacterToken
from src.data.DataObjects.TransformerTextDataObject import TransformerTextDataObject

#from src.models.Transformer.Transformer import Transformer
#from src.models.LossAndMetrics.LossAndMetrics import masked_loss, masked_accuracy, CustomSchedule
#from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, OutputTextCallback
#from src.models.TextGenerators.StandardTransformerGenerator import StandardTransformerGenerator

import os
import tensorflow as tf

#Meta Info
model_name = "W_P_T_S2.0"

#Data hyperparameters
data_soure = "Webscrape"
data_sequencing_len = 1

#Pre processing hyperparameters
token_seqence_length = 75

#File path values
root_dir = os.path.abspath("./")
processed_data = os.path.join(root_dir, "data", "processed")

#************************DATA************************

#Tokens (Text to sequence)
context_token = WordpieceToken(vocab_size=8000, sequence_len=token_seqence_length)
content_token = WordpieceToken(vocab_size=8000, sequence_len=token_seqence_length)


#Data objects
base_data_object = TransformerTextDataObject(context_sequencer=context_token, content_sequencer=content_token
                                                 , context_len=token_seqence_length, content_len=token_seqence_length
                                                 , data_loader = complete_transformer_retriever
                                                 , base_path=processed_data, data_source=data_soure, data_sequencing_len=data_sequencing_len
                                                 , set_suffix="base")



validation_data_object = TransformerTextDataObject(context_sequencer=base_data_object.context_sequencer, content_sequencer=base_data_object.content_sequencer
                                                 , context_len=token_seqence_length, content_len=token_seqence_length
                                                 , data_loader = complete_transformer_retriever, init_tokens=False
                                                 , base_path=processed_data, data_source=data_soure, data_sequencing_len=data_sequencing_len
                                                 , set_suffix="val")


























validation_dataset = validation_data_object.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)


























tester= StandardTransformerGenerator(input_str="hello this is my brother, he is a good person", source_model=trans_inst, output_len=token_seqence_length
                                     ,context_sequencer=training_data_object.context_sequencer, content_sequencer=training_data_object.content_sequencer)

tester.generate_output()


trans_inst_2 = tester.source_model

model_name_save = "W_P_T_M1.0"
model_path = os.path.join(root_dir, "models", model_name_save, "model_file")

print(model_path)


trans_inst_2.save(model_path, save_format="tf")