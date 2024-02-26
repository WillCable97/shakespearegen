import os
import tensorflow as tf
from src.data.TextToToken.MyTfTokenizer import MyTfToken
from src.data.DataLoaders import get_webscrape_data
from src.data.DataObjects.TextDataObject import TransformerTextDataObject
from src.models.Transformer.Transformer import Transformer
from src.models.LossAndMetrics import masked_loss, CustomSchedule, masked_accuracy
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback


#Project details
project_directory = os.path.abspath("./")
path_to_data_folder = os.path.join(project_directory, "data/processed/webdata")
context_token = MyTfToken()
content_token = MyTfToken()

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
                                        , context_len=sequence_length, content_len=sequence_length, use_bookmark_tokens=True
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



from src.data.TextToToken.TextToToken import TextToToken
import numpy as np
import keras
class TextGenerator(keras.callbacks.Callback):
    def __init__(self,input_model: tf.keras.models.Model, context_lenght: int, content_length: int
                 ,context_sequencer: TextToToken, content_sequencer: TextToToken, input_str: str):
        self.input_model = input_model
        self.context_lenght = context_lenght
        self.content_length = content_length
        self.context_sequencer = context_sequencer
        self.content_sequencer = content_sequencer
        self.input_str = input_str
        self.context_vector = self.create_context_vector()
        print(self.context_vector)

    def create_context_vector(self): #Need to check here is the input is higher than the max len -2
        corrected_string = f"starttoken {self.input_str} endtoken"
        token_seq = self.context_sequencer.tokenise([corrected_string])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(token_seq, maxlen=self.context_lenght, padding='post')
        return np.array(padded_seq)
    
    def generate_output(self):
        geerated_tokens = []
        content_vector = self.content_sequencer.tokenise(["starttoken"])
        content_vector = np.array(content_vector)
        end_token = self.content_sequencer.tokenise(["endtoken"])[0][0]
        token_index = 1  

        for i in range(self.content_length - 2):
            if i >= 20: break #Runtime overload
            content_padded = tf.keras.preprocessing.sequence.pad_sequences(content_vector, maxlen=self.content_length)
            model_output = self.input_model([self.context_vector, content_padded])
            index_output = model_output[0][token_index]
            found_token = tf.argmax(index_output)
            if found_token == end_token:
                print("\n REACHED TERMINATION")
                break 

            content_vector = np.hstack((content_vector, [[found_token]]))
            token_index += 1
        return ' '.join(self.content_sequencer.detokenise(content_vector)[1:])


            
    def on_epoch_end(self, epoch, logs=None):
        self.input_model = trans_inst
        txt = self.generate_output()
        print(f"\ngenerated text:\n{txt}\n")


Test = TextGenerator(input_model=trans_inst, context_lenght=sequence_length
                     , content_length=sequence_length, context_sequencer=my_data_set.context_sequencer
                     , content_sequencer=my_data_set.content_sequencer
                     , input_str="Hello this is my good friend, he is a good guy")



trans_inst.fit(training_dataset, epochs=10, callbacks=[my_csv_callback, my_checkpoint_callback, Test], )
