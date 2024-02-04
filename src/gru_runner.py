from data.data import CharacterIndexing, DataObject
from models.GRU_model import GRU_model
import tensorflow as tf
import os

char_indexing_obj = CharacterIndexing()
data_object_initial = DataObject(hugging_face_name="tiny_shakespeare"
                                 ,index_obj=char_indexing_obj
                                 ,sequence_length=100
                                 ,batch_size=64
                                 ,buffer_size=10000)


vocab_size = len(data_object_initial.text_to_int())
gru_model = GRU_model(vocab_size, 256, 1024, 64)


#Define some things for compilation
def loss_func(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


gru_model.compile(optimizer='adam', loss=loss_func)


csv_dir = './src/models/callbacks/gru/csv'
checkpoint_dir = './src/models/callbacks/gru/checkpoint'


checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
csv_prefix = os.path.join(csv_dir, "csv_tracker")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


csv_callback = tf.keras.callbacks.CSVLogger(csv_prefix)

EPOCHS=10
history = gru_model.fit(data_object_initial.final_dataset, epochs=10, callbacks=[checkpoint_callback, csv_callback])

