import os 

def read_in_text_files(*args): #list of file paths of text files
    return_lists = []
    for x in args:
        return_lists.append(parse_single_file(x))

    return return_lists


def parse_single_file(file_path:str):
    file = open(file_path, "r", encoding="utf8")
    text = file.read()
    data = text.split("[SEQ_SPLITTER]")
    return data

def complete_transformer_retriever(base_path: str, data_source: str, data_sequencing_len:int, set_suffix = "train"):
    context_path = os.path.join(base_path, data_source, f"Seq{data_sequencing_len}", f"context_{set_suffix}.txt")
    content_path = os.path.join(base_path, data_source, f"Seq{data_sequencing_len}", f"content_{set_suffix}.txt")
    return read_in_text_files(context_path, content_path)

def complete_single_emb_retriever(base_path: str, data_source: str, data_sequencing_len:int, set_suffix = "train"):
    text_path = os.path.join(base_path, data_source, f"Seq{data_sequencing_len}", f"{set_suffix}.txt")
    return read_in_text_files(text_path)[0][:-1]

