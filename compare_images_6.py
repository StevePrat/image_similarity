# -*- coding: utf-8 -*
# compare between 2 different folders
# use pandas DataFrame instead of dictionaries when possible
# input file includes shop_id, compare only cross-shop

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

def csv_to_df(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name, names=['grass_region','shop_id','item_id','img_id'])
    df = df.drop(columns='grass_region')
    df: pd.DataFrame = df[df['img_id'].map(lambda x: set(str(x).lower()).issubset(set('0123456789abcdef')))]
    df = df.astype({
        'shop_id': int,
        'item_id': int,
        'img_id': str
    })

    return df

def compare_intra_df(to_compare_df: pd.DataFrame) -> pd.DataFrame:
    img_hash_df = get_stored_hashes(to_compare_df['img_id'])
    to_compare_df = to_compare_df.merge(
        right=img_hash_df,
        on='img_id'
    )

    shop_pairs = list(itertools.combinations(to_compare_df['shop_id'].unique(), 2))

    df = pd.DataFrame(
        columns=['shop_id_1', 'shop_id_2'],
        data=([shop_1, shop_2] for shop_1, shop_2 in shop_pairs)
    )
    df = df.merge(
        right=to_compare_df,
        left_on='shop_id_1',
        right_on='shop_id'
    ).rename(
        columns={
            'item_id': 'item_id_1',
            'img_id': 'img_id_1',
            'hash': 'hash_1'
        }
    ).merge(
        right=to_compare_df,
        left_on='shop_id_2',
        right_on='shop_id'
    ).rename(
        columns={
            'item_id': 'item_id_2',
            'img_id': 'img_id_2',
            'hash': 'hash_2'
        }
    )

    df['difference'] = df['hash_1'] - df['hash_2']
    df = df.drop(columns=['hash_1', 'hash_2'])

    if THRESHOLD is not None:
        df = df[df['difference'] <= THRESHOLD]

    if len(df) == 0:
        return df
    
    write_result_file(df)
    return df

def compare_inter_df(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    assert len(dfs) == 2, 'length of sequence of DataFrames must be 2'
    df_1, df_2 = dfs

    img_hash_df_1 = get_stored_hashes(df_1['img_id'])
    img_hash_df_2 = get_stored_hashes(df_2['img_id'])

    df_1 = df_1.merge(
        right=img_hash_df_1,
        on='img_id'
    )
    df_2 = df_2.merge(
        right=img_hash_df_2,
        on='img_id'
    )

    shop_pairs = list(itertools.product(df_1['shop_id'], df_2['shop_id']))

    df = pd.DataFrame(
        columns=['shop_id_1', 'shop_id_2'],
        data=([shop_1, shop_2] for shop_1, shop_2 in shop_pairs if shop_1 != shop_2)
    )

    df = df.merge(
        right=df_1,
        left_on='shop_id_1',
        right_on='shop_id'
    ).drop(
        columns='shop_id'
    ).rename(
        columns={
            'item_id': 'item_id_1',
            'img_id': 'img_id_1',
            'hash': 'hash_1'
        }
    ).merge(
        right=df_2,
        left_on='shop_id_2',
        right_on='shop_id'
    ).drop(
        columns='shop_id'
    ).rename(
        columns={
            'item_id': 'item_id_2',
            'img_id': 'img_id_2',
            'hash': 'hash_2'
        }
    )

    df['difference'] = df['hash_1'] - df['hash_2']
    df = df.drop(columns=['hash_1', 'hash_2'])

    if THRESHOLD is not None:
        df = df[df['difference'] <= THRESHOLD]
    
    if len(df) == 0:
        return df
    
    write_result_file(df)
    return df

def df_pair_generator(file_names_1: Iterable[str], file_names_2: Iterable[str], split_size: int) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    file_names_1 = list(file_names_1)
    file_names_1.sort()
    file_names_2 = list(file_names_2)
    file_names_2.sort()
    for fn_1 in file_names_1:
        print('Opening', fn_1)
        df_1 = csv_to_df(fn_1)
        df_1 = df_1.sort_values('img_id')
        number_of_splits_1 = len(df_1) // split_size
        for fn_2 in file_names_2:
            print('Opening', fn_2)
            df_2 = csv_to_df(fn_2)
            df_2 = df_2.sort_values('img_id')
            number_of_splits_2 = len(df_2) // split_size
            print('Current pair', fn_1, fn_2)
            for sub_df_1 in np.array_split(df_1, number_of_splits_1):
                for sub_df_2 in np.array_split(df_2, number_of_splits_2):
                    yield sub_df_1, sub_df_2

def main() -> None:
    global counter
    counter = Value(ctypes.c_uint64, 0)

    print('Starting', datetime.datetime.now())
    file_names_1 = [FOLDER_1 + fn for fn in os.listdir(FOLDER_1)]
    file_names_2 = [FOLDER_2 + fn for fn in os.listdir(FOLDER_2)]

    with Pool(PARALLEL_PROCESSES) as pool:
        print('Starting cross comparison')
        for i, difference_df in enumerate(pool.imap(compare_inter_df, df_pair_generator(file_names_1, file_names_2, SPLIT_SIZE))):
            print('[{}] Part {} done'.format(datetime.datetime.now(), i+1))

    print('Done', datetime.datetime.now())

if __name__ == '__main__':
    main()