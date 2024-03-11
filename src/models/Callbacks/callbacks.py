import os 
import keras
from keras.callbacks import CSVLogger, ModelCheckpoint
from src.models.TextGenerators.TextGenerator import TextGenerator

#seperate out dir creations ???

def path_to_model_saves(base_path, input_model_name):
    path_to_model_saves = os.path.join(base_path, f"models/{input_model_name}")
    if not os.path.exists(path_to_model_saves): os.makedirs(path_to_model_saves)
    return path_to_model_saves

def csv_callback(base_path, input_model_name):
    full_path = path_to_model_saves(base_path, input_model_name)
    full_path = os.path.join(full_path, "csv_tracker")
    if not os.path.exists(full_path): os.makedirs(full_path)
    csv_prefix = os.path.join(full_path, "csv_tracker")
    csv_callback = CSVLogger(csv_prefix)
    return csv_callback


def checkpoint_callback(base_path, input_model_name, period):
    full_path = path_to_model_saves(base_path, input_model_name)
    full_path = os.path.join(full_path, "checkpoint_tracker")
    if not os.path.exists(full_path): os.makedirs(full_path)
    checkpoint_prefix = os.path.join(full_path, "ckpt_{epoch}.weights.h5",)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_prefix
        ,save_weights_only=True
        ,save_freq='epoch')
    return checkpoint_callback


class OutputTextCallback(keras.callbacks.Callback):
    def __init__(self, input_generator: TextGenerator, base_path:str
                 , input_model_name:str , file_name = 'text_outputs.txt'):
        super().__init__()
        self.input_generator = input_generator
        self.base_path = base_path
        self.input_model_name = input_model_name
        self.file_name = file_name
        self.txt_generated = ''

    def on_epoch_end(self, epoch, logs=None):
        self.input_generator.source_model.set_weights(self.model.get_weights())
        txt = self.input_generator.generate_output()
        print(f"\ngenerated text:  {txt}\n")
        self.txt_generated += f"{epoch + 1}: {txt}\n"
        self.write_output_to_file()

    def write_output_to_file(self):
        full_path = path_to_model_saves(self.base_path, self.input_model_name)
        full_path = os.path.join(full_path, "text_tracker")
        if not os.path.exists(full_path): os.makedirs(full_path)
        file_path = os.path.join(full_path, self.file_name)
        file = open(file_path, 'w+')
        file.write(self.txt_generated)
    





