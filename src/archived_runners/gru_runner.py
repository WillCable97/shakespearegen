from data.data import CharacterIndexing, DataObject
from models.GRU_model import GRU_model
import tensorflow as tf
import os



sequence_lengths = [50,100,150,200]

EPOCHS=20

#Define some things for compilation
def loss_func(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

char_indexing_obj = CharacterIndexing()

for sequence in sequence_lengths:
    data_object_initial = DataObject(hugging_face_name="tiny_shakespeare"
                                    ,index_obj=char_indexing_obj
                                    ,sequence_length=sequence
                                    ,batch_size=64
                                    ,buffer_size=10000)
    
    vocab_size = len(data_object_initial.text_to_int())
    gru_model = GRU_model(vocab_size, 256, 1024, 64)
    gru_model.compile(optimizer='adam', loss=loss_func)


    model_base_path = f"./src/models/callbacks/gru/sequence_{sequence}"


    checkpoint_path = os.path.join(model_base_path, "checkpoint")#, "ckpt_{epoch}")
    csv_path = os.path.join(model_base_path, "csv")#, "csv_tracker")
    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt_{epoch}")
    csv_prefix = os.path.join(csv_path, "csv_tracker")
    os.makedirs(checkpoint_path)
    os.makedirs(csv_path)

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)


    csv_callback = tf.keras.callbacks.CSVLogger(csv_prefix)
    
    history = gru_model.fit(data_object_initial.final_dataset, epochs=10, callbacks=[checkpoint_callback, csv_callback])


