import os 
from keras.callbacks import CSVLogger, ModelCheckpoint



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


def checkpoint_callback(base_path, input_model_name):
    full_path = path_to_model_saves(base_path, input_model_name)
    full_path = os.path.join(full_path, "checkpoint_tracker")
    if not os.path.exists(full_path): os.makedirs(full_path)
    checkpoint_prefix = os.path.join(full_path, "ckpt_{epoch}")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    return checkpoint_callback
