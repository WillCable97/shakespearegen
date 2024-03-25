from src.data.DataLoadersDir.EnhancedText import complete_transformer_retriever
from src.data.TextToToken.GptToken import GptToken
from src.data.DataObjects.TransformerTextDataObject import TransformerTextDataObject
"""
from src.models.Transformer.Transformer import Transformer
from src.models.LossAndMetrics import masked_loss, masked_accuracy, CustomSchedule
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback, OutputTextCallback
from src.models.TextGenerators.StandardTransformerGenerator import StandardTransformerGenerator

import os
import tensorflow as tf
"""

import os


#Meta Info
model_name = "W_P_T_S2.0"

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

#File path values
root_dir = os.path.abspath("./")
processed_data = os.path.join(root_dir, "data", "processed")

#************************DATA************************

#Tokens (Text to sequence)
context_token = GptToken()
content_token = GptToken()

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




#from transformers import GPT2Config
#config = GPT2Config(vocab_size=50258)


#from transformers import TFTrainer, TFTrainingArguments
from transformers import TFGPT2LMHeadModel



# Define the GPT model
model = TFGPT2LMHeadModel.from_pretrained("gpt2")



# Get the ID of the end-of-sequence token
eos_token_id = base_data_object.context_sequencer.eos_token_id

# Set the model's pad_token_id attribute
model.config.pad_token_id = eos_token_id
#model.config.vocab_size = 50259



import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)


model.fit(training_dataset, epochs=3)



"""
from transformers import TFTrainer, TFTrainingArguments
training_args = TFTrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
)
"""










"""

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

tester= StandardTransformerGenerator(input_str="hello this is my brother, he is a good person", source_model=trans_inst, output_len=token_seqence_length
                                     ,context_sequencer=training_data_object.context_sequencer, content_sequencer=training_data_object.content_sequencer)
output_callback = OutputTextCallback(tester, root_dir, model_name)

tester2= StandardTransformerGenerator(input_str="this morning the bird flew to her nest and laid an egg", source_model=trans_inst, output_len=token_seqence_length
                                     ,context_sequencer=training_data_object.context_sequencer, content_sequencer=training_data_object.content_sequencer)
output_callback2 = OutputTextCallback(tester2, root_dir, model_name)

tester3= StandardTransformerGenerator(input_str="last year i went on holidays to another country, it was very relaxing and I would like to go back", source_model=trans_inst, output_len=token_seqence_length
                                     ,context_sequencer=training_data_object.context_sequencer, content_sequencer=training_data_object.content_sequencer)
output_callback3 = OutputTextCallback(tester3, root_dir, model_name)

trans_inst.fit(training_dataset, epochs=epoch_count, validation_data=validation_dataset
               , callbacks=[my_csv_callback, my_checkpoint_callback, output_callback, output_callback2, output_callback3])

"""