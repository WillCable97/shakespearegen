from src.data.DataLoadersDir.EnhancedText import complete_transformer_retriever
from src.data.TextToToken.WordpieceToken import WordpieceToken
from src.data.DataObjects.TransformerTextDataObject import TransformerTextDataObject

from src.models.Transformer.Transformer import Transformer
from src.models.LossAndMetrics import masked_loss, masked_accuracy, CustomSchedule
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, OutputTextCallback
from src.models.TextGenerators.StandardTransformerGenerator import StandardTransformerGenerator

import os
import tensorflow as tf

#Meta Info
model_name = "W_P_T_M1.0"

#Data hyperparameters
data_soure = "Webscrape"
data_sequencing_len = 1

#Pre processing hyperparameters
token_seqence_length = 50
batch_size = 64
buffer_size = 10000

#Model hyperparameters
embedding_dimension = 256
dense_dimension = 512
num_heads = 8
num_att_layers = 4
dropout_rate = 0.1
epoch_count = 40

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


training_data_object = TransformerTextDataObject(context_sequencer=base_data_object.context_sequencer, content_sequencer=base_data_object.content_sequencer
                                                 , context_len=token_seqence_length, content_len=token_seqence_length
                                                 , data_loader = complete_transformer_retriever, init_tokens=False
                                                 , base_path=processed_data, data_source=data_soure, data_sequencing_len=data_sequencing_len)





#Vocab printing
vocab_size_shake = base_data_object.content_vocab
vocab_size_eng = base_data_object.context_vocab
print(f"Shakespeare Vocab: {vocab_size_shake} , English Vocab: {vocab_size_eng}")

training_dataset = training_data_object.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)
validation_dataset = validation_data_object.batch_and_shuffle(batch_size=batch_size,buffer_size=buffer_size)


#************************MODEL COMPILATION************************

#Model definition
trans_inst = Transformer(vocab_size=vocab_size_shake, context_vocab_size=vocab_size_eng
                         , embedding_dimension=embedding_dimension,  dense_dimension=dense_dimension
                         , context_length=token_seqence_length,content_length=token_seqence_length
                         , num_att_layers=num_att_layers, num_heads=num_heads)


learning_rate = CustomSchedule(embedding_dimension)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

trans_inst.compile(optimizer, loss=[masked_loss],metrics=[masked_accuracy])



#************************MODEL CALLBACKS************************

my_csv_callback = csv_callback(root_dir, model_name)
my_checkpoint_callback = checkpoint_callback(root_dir, model_name,5)


tester2= StandardTransformerGenerator(input_str="this morning the bird flew to her nest and laid an egg", source_model=trans_inst, output_len=token_seqence_length
                                     ,context_sequencer=training_data_object.context_sequencer, content_sequencer=training_data_object.content_sequencer)
output_callback2 = OutputTextCallback(tester2, root_dir, model_name)

trans_inst.fit(training_dataset, epochs=epoch_count, validation_data=validation_dataset
               , callbacks=[my_csv_callback, my_checkpoint_callback, output_callback2])




























"""
# Assuming your dataset is named 'dataset'
first_batch = next(iter(training_dataset.take(1)))

# Unpack the elements of the first batch
((context, content), label) = first_batch



print(context[0:1].numpy().tolist())
print(content[0:1].numpy().tolist())
print(label[0:1].numpy().tolist())



print(f"CONTEXT: {training_data_object.context_sequencer.detokenise(context[0:1].numpy().tolist())}")
print(f"CONTENT: {training_data_object.content_sequencer.detokenise(content[0:1].numpy().tolist())}")
print(f"LABEL: {training_data_object.content_sequencer.detokenise(label[0:1].numpy().tolist())}")


"""


