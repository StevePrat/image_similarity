# -*- coding: utf-8 -*
# compare images using multiprocessing pool.imap

from multiprocessing.sharedctypes import SynchronizedBase
import pandas as pd
import imagehash
import itertools
from typing import *
from multiprocessing.pool import Pool
from multiprocessing import Value
import datetime
import optparse
import os

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
    '--folder',
    action="store", dest="folder",
    help="folder containing files to compare", 
    default = None
)

options, args = parser.parse_args()

THRESHOLD: str = options.threshold
if THRESHOLD is not None:
    THRESHOLD = int(THRESHOLD)

PARALLEL_PROCESSES = int(options.parallel_processes)

HOME_PATH = './'

STORED_HASH_FOLDER: str = options.folder
if STORED_HASH_FOLDER is None:
    STORED_HASH_FOLDER = HOME_PATH + 'stored_hash/'
elif not STORED_HASH_FOLDER.startswith('/'):
    STORED_HASH_FOLDER = HOME_PATH + STORED_HASH_FOLDER
if not STORED_HASH_FOLDER.endswith('/'):
    STORED_HASH_FOLDER += '/'

OUTPUT_FOLDER = HOME_PATH + 'comparison_result/'
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

def get_stored_hashes(img_id_list: Iterable[str]) -> Dict[str, imagehash.ImageHash]:
    img_id_grp = get_img_id_grp(img_id_list)
    img_hash_map = {k: None for k in img_id_list}
    
    for img_id_head, img_id_subset in img_id_grp.items():
        file_path = STORED_HASH_FOLDER + img_id_head + '.csv'
        if not os.path.exists(file_path):
            continue
        
        stored_df = pd.read_csv(file_path, names=['img_id', 'hash'])
        filtered_df = stored_df[stored_df['img_id'].isin(img_id_subset)]

        if len(filtered_df) == 0:
            continue

        img_hashes_to_add = filtered_df.apply(lambda x: {x['img_id']: imagehash.hex_to_hash(str(x['hash']))}, axis=1)
        for d in img_hashes_to_add:
            img_hash_map.update(d)

    return img_hash_map

def difference_map_to_csv(difference_map: Dict[Tuple[str, str], int], file_name: str) -> None:
    df = pd.DataFrame(
        columns=['img_id_1','img_id_2','difference'],
        data=([img_id_1, img_id_2, difference] for (img_id_1, img_id_2), difference in difference_map.items())
    )
    df.to_csv(file_name, index=False)

def compare_intra_map(img_hash_map: Dict[str, imagehash.ImageHash]) -> Dict[Tuple[str, str], int]:
    img_id_list = list(img_hash_map.keys())
    img_id_list.sort()
    comparison_result: Dict[Tuple[str, str], int] = {}
    for img_id_1, img_id_2 in itertools.combinations(img_id_list, 2):
        img_hash_1 = img_hash_map.get(img_id_1)
        img_hash_2 = img_hash_map.get(img_id_2)
        if (img_hash_1 is None) or (img_hash_2 is None):
            continue
        
        difference: int = img_hash_1 - img_hash_2
        comparison_result.update({(img_id_1, img_id_2): difference})
    
    return comparison_result

def compare_intra_file(file_name: str) -> Dict[Tuple[str, str], int]:
    img_hash_map: Dict[str, imagehash.ImageHash] = {}
    df = pd.read_csv(file_name, names=['img_id', 'hash'])
    df.apply(lambda x: img_hash_map.update({x['img_id']: imagehash.hex_to_hash(str(x['hash']))}), axis=1)
    result = compare_intra_map(img_hash_map)
    if THRESHOLD is not None:
        result = {k: v for k, v in result.items() if v <= THRESHOLD}

    with counter.get_lock():
        counter_value = counter.value + 1
        counter.value = counter_value
    
    output_file_path = OUTPUT_FOLDER + OUTPUT_NAME_TEMPL.format(counter_value)
    difference_map_to_csv(result, output_file_path)

    return result

def compare_inter_map(img_hash_map_1: Dict[str, imagehash.ImageHash], img_hash_map_2: Dict[str, imagehash.ImageHash]) -> Dict[Tuple[str, str], int]:
    comparison_result: Dict[Tuple[str, str], int] = {}
    for img_id_1, img_id_2 in itertools.product(img_hash_map_1.keys(), img_hash_map_2.keys()):
        if img_id_1 > img_id_2:
            img_id_1, img_id_2 = img_id_2, img_id_1

        img_hash_1 = img_hash_map_1.get(img_id_1)
        img_hash_2 = img_hash_map_2.get(img_id_2)
        if (img_hash_1 is None) or (img_hash_2 is None):
            continue

        difference: int = img_hash_1 - img_hash_2
        comparison_result.update({(img_id_1, img_id_2): difference})
    
    return comparison_result

def compare_inter_file(file_names: Sequence[str]) -> Dict[Tuple[str, str], int]:
    global counter

    assert len(file_names) == 2, 'length of file name sequence must be 2'
    file_name_1, file_name_2 = file_names
    
    df_1 = pd.read_csv(file_name_1, names=['img_id', 'hash'])
    df_2 = pd.read_csv(file_name_2, names=['img_id', 'hash'])
    img_hash_map_1: Dict[str, imagehash.ImageHash] = {}
    img_hash_map_2: Dict[str, imagehash.ImageHash] = {}
    df_1.apply(lambda x: img_hash_map_1.update({x['img_id']: imagehash.hex_to_hash(str(x['hash']))}), axis=1)
    df_2.apply(lambda x: img_hash_map_2.update({x['img_id']: imagehash.hex_to_hash(str(x['hash']))}), axis=1)
    result = compare_inter_map(img_hash_map_1, img_hash_map_2)
    if THRESHOLD is not None:
        result = {k: v for k, v in result.items() if v <= THRESHOLD}

    with counter.get_lock():
        counter_value = counter.value + 1
        counter.value = counter_value
    
    output_file_path = OUTPUT_FOLDER + OUTPUT_NAME_TEMPL.format(counter_value)
    difference_map_to_csv(result, output_file_path)

    return result

def main() -> None:
    global counter
    counter = Value('B', 0)

    print('Starting', datetime.datetime.now())
    files_to_compare = [STORED_HASH_FOLDER + fn for fn in os.listdir(STORED_HASH_FOLDER)]

    with Pool(PARALLEL_PROCESSES) as pool:
        ### self compare
        print('Starting self comparison')
        for i, difference_map in enumerate(pool.imap(compare_intra_file, files_to_compare)):
            print('[{}] Part {} done'.format(datetime.datetime.now(), i+1))

        ### compare with other files
        print('Starting cross comparison')
        for i, difference_map in enumerate(pool.imap(compare_inter_file, itertools.combinations(files_to_compare, 2))):
            print('[{}] Part {} done'.format(datetime.datetime.now(), i+1))

    print('Done', datetime.datetime.now())

if __name__ == '__main__':
    main()