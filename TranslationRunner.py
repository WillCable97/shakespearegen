from src.data.TokenDataObj import TokenDataObj
from keras.preprocessing.text import Tokenizer
from src.models.Transformer.Transformer import Transformer
from src.models.Callbacks.callbacks import csv_callback, checkpoint_callback
import tensorflow as tf
import os
from urllib.request import urlopen
from bs4 import BeautifulSoup

#Project Details
input_token = Tokenizer()
context_token = Tokenizer()
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




url = "https://www.litcharts.com/shakescleare/shakespeare-translations/all-s-well-that-ends-well/act-1-scene-1"
page = urlopen(url)
html_bytes = page.read()
soup = BeautifulSoup(html_bytes, 'html.parser')


def get_text_from_site(input_soup):
    div_list =["col original-play", "translation-content"]#"col modern-translation"]
    classes_list = ["stage-directions", "acs-character-heading", "speaker-text"]
    all_line_segments = input_soup.find_all("div", {"class": "comparison-row"})
    english_lines =[]
    og_lines =[]
    i=0

    for line_segment in all_line_segments:
        side_by_side = []
        for version in div_list:
            version_div = line_segment.find("div", {"class": version})
            if version_div == None: continue

            version_text_vector = [[text_element.get_text().strip() for text_element in version_div.find_all("p", {"class": type_name})] 
                                   for type_name in classes_list]
            side_by_side.append(version_text_vector)
        i+=1
        if i==1: continue

        if not (side_by_side[0][2] == [] and side_by_side[1][2] == []):
            english_lines.append(''.join(side_by_side[1][2]))
            og_lines.append(''.join(side_by_side[0][2]))

    return english_lines, og_lines

A,B = get_text_from_site(soup)#englsg, og


def straight_return(input):
    return input


english_set = TokenDataObj(tokenizer=context_token)
english_set.iniatiate_with_loader(straight_return, input = A)
english_set.final_dataset(sequence_length=sequence_length, batch_size=batch_size, buffer_size=buffer_size,  padding_required=True, create_label=False)

shake_set = TokenDataObj(tokenizer=input_token)
shake_set.iniatiate_with_loader(straight_return, input = B)
shake_set.final_dataset(sequence_length=sequence_length, batch_size=batch_size, buffer_size=buffer_size,  padding_required=True, create_label=True)

vocab_size_eng = english_set.vocab_size
vocab_size_shake = shake_set.vocab_size

#print(f"{vocab_size_eng}, {vocab_size_shake}")

trans_inst = Transformer(vocab_size=vocab_size_shake, context_vocab_size=vocab_size_eng,embedding_dimension=embedding_dimension
                                            ,sequence_length=sequence_length
                                            ,num_heads=num_heads,dense_dimension=dense_dimension,num_att_layers=num_att_layers)


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


trans_inst.compile("adam", loss=[masked_loss, None],metrics=["accuracy", None])
trans_inst.fit(test_text.final_dataset_obj, epochs=3)

"""

my_csv_callback = csv_callback(project_directory, model_name)
my_checkpoint_callback = checkpoint_callback(project_directory, model_name)

"""

