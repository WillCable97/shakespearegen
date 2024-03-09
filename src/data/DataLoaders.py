import os
import pickle

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

    #all_eng_text = [[all_eng_text[15], all_eng_text[54], all_eng_text[67], all_eng_text[35]] for x in all_eng_text]
    #all_og_text = [[all_og_text[15], all_og_text[54], all_og_text[67], all_og_text[35]] for x in all_og_text]

    #all_eng_text = [a for b in all_eng_text for a in b]
    #all_og_text = [a for b in all_og_text for a in b]


    return all_eng_text[:5000], all_og_text[:5000]


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


from datasets import load_dataset

def get_data_from_hgset(set_name: str, sequence_len: int):
    hg_data = load_dataset(set_name)
    all_text = hg_data["train"]["text"][0]
    seq_count = int(len(all_text)/sequence_len)
    ret_list = []

    for i in range(seq_count):
        seq_string = all_text[i*sequence_len:(i+1)*sequence_len]
        ret_list.append(seq_string)
    
    return ret_list