import os
import pickle
from datasets import load_dataset
import numpy as np

#Get webscrape data lists
def get_webscrape_data(data_path: str):
    all_eng_text = []
    all_og_text =[]

    for dir_eg in os.listdir(data_path):
        path_to_play = os.path.join(data_path, dir_eg)
        
        with open(os.path.join(path_to_play, "english_lines.txt"), "rb") as fp:   # Unpickling
            play_eng_lines = pickle.load(fp)

        with open(os.path.join(path_to_play, "og_lines.txt"), "rb") as fp2:   # Unpickling
            play_og_lines = pickle.load(fp2)

        if len(play_eng_lines) != len(play_og_lines): print("PROBLEMS")

        all_eng_text += play_eng_lines
        all_og_text += play_og_lines

    return all_eng_text, all_og_text#[:5000]

def get_lines_for_backwards_testing(data_path: str, sequence_len: int):
    all_eng_text, _ = get_webscrape_data(data_path=data_path)
    all_eng_text = all_eng_text[:5000]

    entire_text = ' '.join(all_eng_text)
    seq_count = int(len(entire_text)/sequence_len)
    forward_list = []
    back_list = []

    for i in range(seq_count):
        forawrd_string = entire_text[i*sequence_len:(i+1)*sequence_len]
        forward_list.append(forawrd_string)
        back_list.append(forawrd_string[::-1])
    
    return forward_list, back_list




def get_data_from_hgset(set_name: str, sequence_len: int):
    hg_data = load_dataset(set_name)
    all_text = hg_data["train"]["text"][0]
    seq_count = int(len(all_text)/sequence_len)
    ret_list = []

    for i in range(seq_count):
        seq_string = all_text[i*sequence_len:(i+1)*sequence_len]
        ret_list.append(seq_string)
    
    return ret_list


def read_text_data(file_path: str):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines


def read_text_with_sequences(file_path: str, sequence_len: int, len_type = "char"):
    lines = read_text_data(file_path)
    all_lines = ''.join(lines)

    section_seperator =''

    if len_type == "word":
        all_lines = all_lines.replace("\n", " ")
        #all_lines = all_lines.replace("-", " ")
        all_lines = all_lines.split(' ')
        section_seperator  = ' '

    str_len = len(all_lines)
    seq_count = int(str_len/sequence_len)
    ret_list = []

    for i in range(seq_count):
        seq_string = all_lines[i*sequence_len:(i+1)*sequence_len]
        ret_list.append(f'{section_seperator}'.join(seq_string))

    return ret_list#[0:5]


def get_webscrape_data_withends(data_path: str):
    all_eng_text, all_og_tex = get_webscrape_data(data_path)

    all_eng_text = ["* " + x + " *" for x in all_eng_text]
    all_og_tex = ["* " + x + " *" for x in all_og_tex]
    
    return all_eng_text, all_og_tex


def identity_loader(input):
    return input