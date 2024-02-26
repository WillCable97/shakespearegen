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
    return all_eng_text, all_og_text#[:10000]

