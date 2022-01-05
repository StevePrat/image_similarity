# -*- coding: utf-8 -*
# compare between 2 different folders
# use pandas DataFrame instead of dictionaries when possible

from multiprocessing.sharedctypes import SynchronizedBase
import pandas as pd
import numpy as np
import imagehash
import itertools
from typing import *
from multiprocessing.pool import Pool
from multiprocessing import Value
import datetime
import optparse
import os
import ctypes

parser = optparse.OptionParser()

parser.add_option(
    '-p', '--processes', '--process', '--parallel',
    action="store", dest="parallel_processes",
    help="number of parallel process", 
    default = 8
)

parser.add_option(
    '--threshold',
    action="store", dest="threshold",
    help="image hash difference threshold", 
    default = None
)

parser.add_option(
    '--folder-1', '--f1',
    action="store", dest="folder_1",
    help="first folder to compare", 
    default = None
)

parser.add_option(
    '--folder-2', '--f2',
    action="store", dest="folder_2",
    help="second folder to compare",
    default = None
)

parser.add_option(
    '--output_folder', '--output', '--fout',
    action="store", dest="output_folder",
    help="output folder",
    default = 'comparison_result'
)

parser.add_option(
    '--split_size',
    action="store", dest="split_size",
    help="# of images from each folder to compare in a batch",
    default = 1000
)

options, args = parser.parse_args()

THRESHOLD: str = options.threshold
if THRESHOLD is not None:
    THRESHOLD = int(THRESHOLD)

PARALLEL_PROCESSES = int(options.parallel_processes)
SPLIT_SIZE = int(options.split_size)

HOME_PATH = './'
STORED_HASH_FOLDER = HOME_PATH + 'stored_hash/'

def check_folder_name(folder: str) -> str:
    if folder is None:
        return

    if not (folder.startswith('/') or folder.startswith('.')) :
        folder = HOME_PATH + folder
    if not folder.endswith('/'):
        folder += '/'
    return folder

assert options.folder_1 is not None, "missing first folder to compare"
assert options.folder_2 is not None, "missing second folder to compare"

FOLDER_1 = check_folder_name(options.folder_1)
FOLDER_2 = check_folder_name(options.folder_2)

OUTPUT_FOLDER = check_folder_name(options.output_folder)
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

OUTPUT_NAME_TEMPL = 'comparison_result_{}.csv'

counter: SynchronizedBase = None

def get_img_id_head(img_id: str) -> str:
    return img_id[:4]

def get_img_id_grp(img_id_list: Iterable[str]) -> Dict[str, Set[str]]:
    img_id_grp = {get_img_id_head(img_id): set() for img_id in img_id_list}
    for img_id in set(img_id_list):
        img_id_head = get_img_id_head(img_id)
        img_id_grp[img_id_head].add(img_id)
    
    return img_id_grp

def get_stored_hashes(img_id_list: Iterable[str]) -> pd.DataFrame:
    img_id_grp = get_img_id_grp(img_id_list)

    dfs: List[pd.DataFrame] = []
    for img_id_head, img_id_subset in img_id_grp.items():
        file_path = STORED_HASH_FOLDER + img_id_head + '.csv'
        if not os.path.exists(file_path):
            continue
        
        stored_df = pd.read_csv(file_path, names=['img_id', 'hash'])
        filtered_df = stored_df[stored_df['img_id'].isin(img_id_subset)]

        if len(filtered_df) == 0:
            continue

        filtered_df['hash'] = filtered_df['hash'].map(imagehash.hex_to_hash)
        dfs.append(filtered_df)
    
    df = pd.concat(dfs)
    return df

def write_result_file(difference_df: pd.DataFrame) -> None:
    global counter
    
    with counter.get_lock():
        counter_value = counter.value + 1
        counter.value = counter_value
    
    output_file_path = OUTPUT_FOLDER + OUTPUT_NAME_TEMPL.format(counter_value)
    difference_df.to_csv(output_file_path, index=False)

def csv_to_img_id_list(file_name: str) -> List[str]:
    df = pd.read_csv(file_name, names=['grass_region','item_id','img_id'])
    df= df[df['img_id'].map(lambda x: set(str(x).lower()).issubset(set('0123456789abcdef')))]
    img_id_list: np.ndarray = df['img_id'].unique().astype(str)
    img_id_list.sort()
    return img_id_list.tolist()

def compare_intra_df(img_hash_df: pd.DataFrame) -> pd.DataFrame:
    img_id_list = img_hash_df['img_id'].sort_values().tolist()
    img_id_pairs = list(itertools.combinations(img_id_list, 2))

    df = pd.DataFrame(
        columns=['img_id_1', 'img_id_2'],
        data=([img_id_1, img_id_2] for img_id_1, img_id_2 in img_id_pairs)
    )
    df = df.merge(
        right=img_hash_df,
        left_on='img_id_1',
        right_on='img_id'
    ).rename(
        columns={'hash':'hash_1'}
    ).merge(
        right=img_hash_df,
        left_on='img_id_2',
        right_on='img_id'
    ).rename(
        columns={'hash':'hash_2'}
    )

    df['difference'] = df['hash_1'] - df['hash_2']
    df = df.drop(columns=['hash_1', 'hash_2'])

    return df

def compare_inter_df(img_hash_df_1: pd.DataFrame, img_hash_df_2: pd.DataFrame) -> pd.DataFrame:
    img_id_pairs: List[Tuple[str, str]] = []
    for img_id_1, img_id_2 in itertools.product(img_hash_df_1['img_id'], img_hash_df_2['img_id']):
        if img_id_1 > img_id_2:
            img_id_1, img_id_2 = img_id_2, img_id_1

        img_id_pairs.append((img_id_1, img_id_2))

    df = pd.DataFrame(
        columns=['img_id_1', 'img_id_2'],
        data=([img_id_1, img_id_2] for img_id_1, img_id_2 in img_id_pairs)
    )
    df = df.merge(
        right=img_hash_df_1,
        left_on='img_id_1',
        right_on='img_id'
    ).rename(
        columns={'hash':'hash_1'}
    ).merge(
        right=img_hash_df_2,
        left_on='img_id_2',
        right_on='img_id'
    ).rename(
        columns={'hash':'hash_2'}
    )
    df['difference'] = df['hash_1'] - df['hash_2']
    df = df.drop(columns=['hash_1', 'hash_2'])

    return df

def compare_intra_list(img_id_list: Iterable[str]) -> pd.DataFrame:
    img_hash_df = get_stored_hashes(img_id_list)
    result_df = compare_intra_df(img_hash_df)
    if THRESHOLD is not None:
        result_df = result_df[result_df['difference'] <= THRESHOLD]

    if len(result_df) == 0:
        return result_df
    
    write_result_file(result_df)
    return result_df

def compare_inter_list(img_id_lists: Sequence[Iterable[str]]) -> pd.DataFrame:
    assert len(img_id_lists) == 2, 'length of sequence of image id list must be 2'
    img_id_list_1, img_id_list_2 = img_id_lists
    img_hash_df_1 = get_stored_hashes(img_id_list_1)
    img_hash_df_2 = get_stored_hashes(img_id_list_2)
    result_df = compare_inter_df(img_hash_df_1, img_hash_df_2)
    if THRESHOLD is not None:
        result_df = result_df[result_df['difference'] <= THRESHOLD]

    if len(result_df) == 0:
        return result_df
    
    write_result_file(result_df)
    return result_df

def img_id_list_pair_generator(file_names_1: str, file_names_2: str, split_size: int) -> Generator[Tuple[str, str], None, None]:
    for fn_1 in file_names_1:
        print('Opening', fn_1)
        img_id_list_1 = csv_to_img_id_list(fn_1)
        number_of_splits = len(img_id_list_1) // split_size
        for fn_2 in file_names_2:
            print('Opening', fn_2)
            img_id_list_2 = csv_to_img_id_list(fn_2)
            print('Current pair', fn_1, fn_2)
            for img_id_sublist_1 in np.array_split(img_id_list_1, number_of_splits):
                for img_id_sublist_2 in np.array_split(img_id_list_2, number_of_splits):
                    yield img_id_sublist_1, img_id_sublist_2

def main() -> None:
    global counter
    counter = Value(ctypes.c_uint64, 0)

    print('Starting', datetime.datetime.now())
    file_names_1 = [FOLDER_1 + fn for fn in os.listdir(FOLDER_1)]
    file_names_2 = [FOLDER_2 + fn for fn in os.listdir(FOLDER_2)]

    with Pool(PARALLEL_PROCESSES) as pool:
        print('Starting cross comparison')
        for i, difference_df in enumerate(pool.imap(compare_inter_list, img_id_list_pair_generator(file_names_1, file_names_2, SPLIT_SIZE))):
            print('[{}] Part {} done'.format(datetime.datetime.now(), i+1))

    print('Done', datetime.datetime.now())

if __name__ == '__main__':
    main()