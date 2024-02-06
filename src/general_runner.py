import sys
from data.data import CharacterIndexing, DataObject
from models.GRU_model import GRU_model
from models.RNN_model import RNN_model
from models.LSTM_model import LSTM_model
import tensorflow as tf
import os


#System arguments
input_args = sys.argv
model_name = input_args[1]
EPOCHS = int(input_args[2])
sequence_lengths = [50,100,150,200]

#Create function dict
function_dict = {"rnn": RNN_model, "gru": GRU_model, "lstm": LSTM_model}

#Define los function
def loss_func(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#Indexing object
char_indexing_obj = CharacterIndexing()

#File Path fetchers
def get_base_path(model_name: str, sequence: int):
    return  f"./src/models/callbacks/{model_name}/sequence_{sequence}"

def checkpoints_path(model_name: str, sequence: int):
    return os.path.join(get_base_path(model_name, sequence), "checkpoint")

def csv_path(model_name: str, sequence: int):
    return os.path.join(get_base_path(model_name, sequence), "csv")

#Data and model fetching
def fetch_data_seuqnce(sequnce: int): 
    data_object = DataObject(hugging_face_name="tiny_shakespeare"
                            ,index_obj=char_indexing_obj
                            ,sequence_length=sequnce
                            ,batch_size=64
                            ,buffer_size=10000)   
    return data_object, len(data_object.text_to_int()) #Return the vocab size along with the object


def epoch_num_from_path(input_path: str):
    file_name = os.path.basename(input_path)
    file_name = file_name.replace("ckpt_","")
    return int(file_name)


def env_init(model_name: str, sequence: int):
    #File paths and and file objects
    path_to_checkpoints = checkpoints_path(model_name, sequence)
    path_to_csv = csv_path(model_name, sequence)
    model_init = os.path.exists(path_to_checkpoints)


    #Model weight loading
    if model_init:
        recent_check = tf.train.latest_checkpoint(path_to_checkpoints)
        return [path_to_checkpoints, path_to_csv, model_init
                , recent_check, epoch_num_from_path(recent_check)]

    os.makedirs(path_to_checkpoints)
    os.makedirs(path_to_csv)
    return [path_to_checkpoints, path_to_csv, model_init, "", 1]


def fit_model(sequence: int):
    path_to_checkpoints,path_to_csv,model_init,last_check,c_epoch_num = env_init(model_name, sequence)
    data_object, vocab_size= fetch_data_seuqnce(sequence)

    model_obj = function_dict[model_name](vocab_size, 256, 1024, 64)
    model_obj.compile(optimizer='adam', loss=loss_func)

    if model_init: 
        model_obj.load_weights(last_check).expect_partial()
        model_obj.build(tf.TensorShape([64, None]))

    checkpoint_prefix = os.path.join(path_to_checkpoints, "ckpt_{epoch}")
    csv_prefix = os.path.join(path_to_csv, "csv_tracker")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    csv_callback = tf.keras.callbacks.CSVLogger(csv_prefix)

    history = model_obj.fit(data_object.final_dataset, epochs=c_epoch_num + EPOCHS, initial_epoch=c_epoch_num, callbacks=[checkpoint_callback, csv_callback])

for sequence in sequence_lengths: fit_model(sequence)