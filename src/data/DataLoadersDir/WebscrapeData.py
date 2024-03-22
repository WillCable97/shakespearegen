import pickle
import os


def base_webscrape(data_path: str): #All english text: All shakespeare text
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

    return all_eng_text, all_og_text

def base_webscrape_with_ends(data_path: str):
    all_eng_text, all_og_tex = base_webscrape(data_path)

    all_eng_text = ["* " + x + " *" for x in all_eng_text]
    all_og_tex = ["* " + x + " *" for x in all_og_tex]
    
    return all_eng_text, all_og_tex

def overlap_with_ends(data_path: str):
    all_eng_text, all_og_tex = base_webscrape(data_path)

    updated_eng = []
    updated_og = []

    #print(all_eng_text[0])

    for i in range(len(all_eng_text) - 3):
        full_str_eng = ' '.join(all_eng_text[i:i+3])
        full_str_og = ' '.join(all_og_tex[i:i+3])

        updated_eng.append(f"* {full_str_eng} *")
        updated_og.append(f"* {full_str_og} *")


        #file = open("./1.txt", 'w+')
        #file.writelines(updated_eng)

        #file = open("./2.txt", 'w+')
        #file.writelines(updated_eng)


    return updated_eng, updated_og



def training_set(input_datafetcher, data_path:str, training_proportion: float):
    eng, shak = input_datafetcher(data_path)
    number_to_select = int(len(eng) * training_proportion)
    return eng[:number_to_select], shak[:number_to_select]

def val_set(input_datafetcher, data_path:str, val_proportion: float):
    eng, shak = input_datafetcher(data_path)
    number_to_select = int(len(eng) * val_proportion)
    return eng[(-1 * number_to_select):], shak[(-1 * number_to_select):]




