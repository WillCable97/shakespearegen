from data.data import CharacterIndexing, DataObject
from models.GRU_model import GRU_model
from models.RNN_model import RNN_model
from models.LSTM_model import LSTM_model
import tensorflow as tf
import os


def loss_func(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

char_indexing_obj = CharacterIndexing()

data_object_initial = DataObject(hugging_face_name="tiny_shakespeare"
                                ,index_obj=char_indexing_obj
                                ,sequence_length=100
                                ,batch_size=64
                                ,buffer_size=10000)

vocab_size = len(data_object_initial.text_to_int())
gru_model = GRU_model(vocab_size, 256, 1024, 1)
gru_model.compile(optimizer='adam', loss=loss_func)


checkpoint_base = "./src/models/callbacks/gru/sequence_100/checkpoint"
gru_model.load_weights(tf.train.latest_checkpoint(checkpoint_base)).expect_partial()
gru_model.build(tf.TensorShape([1, None]))


text_to_int = data_object_initial.text_to_int()
int_to_text = data_object_initial.int_to_text()


def generate_text(model, start_string):  
  
    num_generate = 1000
    input_eval = [text_to_int[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

   
    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(int_to_text[predicted_id])

    return (start_string + ''.join(text_generated))


eg_text = "the man went"
print(generate_text(gru_model, eg_text))

