# -*- coding: utf-8 -*
# compare images (single process)

import pandas as pd
import imagehash
import itertools
from typing import *
from multiprocessing.pool import Pool
import datetime
import optparse
import os
from collections import deque

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

OUTPUT_NAME = 'comparison_result.csv'

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
    return compare_intra_map(img_hash_map)

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

def compare_inter_file(file_name_1: str, file_name_2: str) -> Dict[Tuple[str, str], int]:
    df_1 = pd.read_csv(file_name_1, names=['img_id', 'hash'])
    df_2 = pd.read_csv(file_name_2, names=['img_id', 'hash'])
    img_hash_map_1: Dict[str, imagehash.ImageHash] = {}
    img_hash_map_2: Dict[str, imagehash.ImageHash] = {}
    df_1.apply(lambda x: img_hash_map_1.update({x['img_id']: imagehash.hex_to_hash(str(x['hash']))}), axis=1)
    df_2.apply(lambda x: img_hash_map_2.update({x['img_id']: imagehash.hex_to_hash(str(x['hash']))}), axis=1)
    return compare_inter_map(img_hash_map_1, img_hash_map_2)

def file_for_intra_compare_generator(file_name_list: Iterable[str], files_per_yield: int) -> Generator[List[str], None, None]:
    file_queue = deque(file_name_list)
    while len(file_queue) >= files_per_yield:
        yield [file_queue.popleft() for _ in range(files_per_yield)]
    else:
        yield [file_queue.popleft() for _ in range(len(file_queue))]
    
def files_for_inter_compare_generator(file_name_list: Iterable[str], combinations_per_yield: str) -> Generator[List[Tuple[str,str]], None, None]:
    combination_queue = deque(itertools.combinations(file_name_list, 2))
    while len(combination_queue) >= combinations_per_yield:
        yield [combination_queue.popleft() for _ in range(combinations_per_yield)]
    else:
        yield [combination_queue.popleft() for _ in range(len(combination_queue))]

def main() -> None:
    print('Starting', datetime.datetime.now())
    files_to_compare = [STORED_HASH_FOLDER + fn for fn in os.listdir(STORED_HASH_FOLDER)]
    
    comparison_result: Dict[Tuple[str, str], int] = {}

    with Pool(PARALLEL_PROCESSES) as pool:
        ### self compare
        print('Starting self comparison')
        for i, file_list in enumerate(file_for_intra_compare_generator(files_to_compare, PARALLEL_PROCESSES)):
            intra_comparison_result_list = pool.map(compare_intra_file, file_list)
            if THRESHOLD is not None:
                intra_comparison_result_list = [{k: v for k, v in d.items() if v <= THRESHOLD} for d in intra_comparison_result_list]
            [comparison_result.update(d) for d in intra_comparison_result_list]
            print('[{}] Part {} done'.format(datetime.datetime.now(), i+1))

        ### compare with other files
        print('Starting cross comparison')
        for i, combination_list in enumerate(files_for_inter_compare_generator(files_to_compare, PARALLEL_PROCESSES)):
            inter_comparison_result_list = pool.starmap(compare_inter_file, combination_list)
            if THRESHOLD is not None:
                inter_comparison_result_list = [{k: v for k, v in d.items() if v <= THRESHOLD} for d in inter_comparison_result_list]
            [comparison_result.update(d) for d in inter_comparison_result_list]
            print('[{}] Part {} done'.format(datetime.datetime.now(), i+1))
    
    print('Combining all results')
    comparison_result_df = pd.DataFrame(
        columns=['img_id_1','img_id_2','difference'],
        data=([img_id_1, img_id_2, difference] for (img_id_1, img_id_2), difference in comparison_result.items())
    )
    comparison_result_df.to_csv(HOME_PATH + OUTPUT_NAME, index=False)

    print('Done', datetime.datetime.now())

if __name__ == '__main__':
    main()