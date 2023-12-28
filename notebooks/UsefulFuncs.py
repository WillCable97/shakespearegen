from pathlib import Path


def root_folder() -> Path:
    return Path(__file__).parents[1]

def src_folder() -> Path:
    return Path.joinpath(root_folder(), 'src')

def data_folder() -> Path: 
    return Path.joinpath(root_folder(), 'data')

def gen_path_from_base(sub_dirs: list) -> Path:
    ret_folder=root_folder() 

    for sub_dir in sub_dirs:
        ret_folder=ret_folder.joinpath(sub_dir)
    
    return ret_folder