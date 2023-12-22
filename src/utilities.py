import pickle
from pathlib import Path
from typing import Union
def save_obj(obj, save_path: Union[str, Path]) -> None:
    '''
    save any python object to disc
    :param obj: any python object
    :param save_dir: file path in str or pathlib.Path
    :return:
    '''
    if type(save_path) is str:
        save_dir = Path(save_path)
    with open(save_path, 'wb') as obj_file:
        # Step 3
        pickle.dump(obj, obj_file)
    return

def load_obj(file_path: Union[str, Path]):
    '''
    load python object
    :param file_path: python object file path in str or pathlib.Path
    :return: python object
    '''
    if type(file_path) is str:
        file_path = Path(file_path)
    with open(file_path, 'rb') as obj_file:
        # Step 3
        obj = pickle.load(obj_file)
    return obj