import TransformerEnhancedData
import os
import tensorflow as tf

model_name = TransformerEnhancedData.model_name
trans_inst = TransformerEnhancedData.trans_inst
training_dataset = TransformerEnhancedData.training_dataset
validation_dataset = TransformerEnhancedData.validation_dataset



root_dir = os.path.abspath("./")



"""
def load_model(model_name: str):
    model_path = os.path.join(root_dir, "models", model_name, "model_file")
    return tf.keras.models.load_model(model_path)

"""

def load_weight(model_name: str, epoch: int):
    path_to_model_saves = os.path.join(root_dir, f"models/{model_name}")
    path_to_checkpoint = os.path.join(path_to_model_saves, "checkpoint_tracker", f"ckpt{epoch}.weights.h5")
    trans_inst.load_weights(path_to_checkpoint)





