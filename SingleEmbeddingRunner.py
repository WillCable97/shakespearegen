from src.data.TokenDataObj import TokenDataObj
from tensorflow.keras.preprocessing.text import Tokenizer
from src.models.Transformer.SingleEmbeddedTransformer import SingleEmbeddedTransformer
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback
import tensorflow as tf
import os

#Project Details
path_to_text_file = "./data/processed/linetext.txt"
input_token = Tokenizer()
project_directory = os.path.abspath("./")
model_name = "SingleEmbTransformerV1"

sequence_length = 80
batch_size = 64
buffer_size = 10000
embedding_dimension = 256
dense_dimension = 256
num_heads = 2
num_att_layers = 2
dropout_rate = 0.1


def text_to_array(input_path: str) -> list:
    file = open(input_path, 'r')
    return file.readlines()[:5000]


#Data 
test_text = TokenDataObj(tokenizer=input_token)
test_text.iniatiate_with_loader(text_to_array, input_path = path_to_text_file)
test_text.final_dataset(sequence_length=sequence_length, batch_size=batch_size, buffer_size=buffer_size,  padding_required=True)

single_emb_trans = SingleEmbeddedTransformer(vocab_size=test_text.vocab_size+1,embedding_dimension=embedding_dimension
                                            ,sequence_length=sequence_length
                                            ,num_heads=num_heads,dense_dimension=dense_dimension,num_att_layers=num_att_layers)

vocab_size = test_text.vocab_size


#Training Metrics
def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss



my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name)


single_emb_trans.compile("adam", loss=[masked_loss, None],metrics=["accuracy", None])
single_emb_trans.fit(test_text.final_dataset_obj, epochs=3, callbacks=[my_csv_callback, my_checkpoint_callback])


