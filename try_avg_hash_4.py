import asyncio
import enum
from multiprocessing.sharedctypes import SynchronizedArray
from PIL import Image
import imagehash
import requests
import io
import itertools
import aiohttp
from typing import *
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import Array
import datetime
import optparse
import os
import pandas as pd
import numpy as np
from collections import deque

parser = optparse.OptionParser()

parser.add_option(
    '-p', '--processes', '--process', '--parallel',
    action="store", dest="parallel_processes",
    help="number of parallel process", 
    default = 8
)

parser.add_option(
    '--split-size',
    action="store", dest="split_size",
    help="size of each split of task", 
    default = 80
)

parser.add_option(
    '--reg', '--region', '--country',
    action="store", dest="reg",
    help="2-letter country code", 
    default = None
)

parser.add_option(
    '--dir', '--folder',
    action="store", dest="folder",
    help="location of files to check", 
    default = None
)

options, args = parser.parse_args()

DOMAIN_MAP = {
    'SG': '.sg',
    'MY': '.com.my',
    'TH': '.co.th',
    'TW': '.tw',
    'ID': '.co.id',
    'VN': '.vn',
    'PH': '.ph',
    'BR': '.com.br',
    'MX': '.com.mx',
    'CO': '.com.co',
    'CL': '.cl',
    'PL': '.pl',
    'FR': '.fr',
    'ES': '.es',
    'IN': '.in'
}

HOME_PATH = './'
STORED_HASH_FOLDER = HOME_PATH + 'stored_hash/'
TO_CHECK_FOLDER: str = HOME_PATH + options.folder
if not TO_CHECK_FOLDER.endswith('/'): TO_CHECK_FOLDER += '/'
REGION = options.reg

PARALLEL_PROCESSES = int(options.parallel_processes)
SPLIT_SIZE = int(options.split_size)

synchronizing_array: SynchronizedArray = None

async def read_response(response: aiohttp.ClientResponse, extra_info: Any = None):
    try:
        if response.content_type.startswith('image'):
            return await response.read()
        else:
            return None
    except Exception as e:
        print('Failed to read response')
        print(response)
        print(extra_info)
        return None

async def get_images_async(url_list: List[str], loop=None) -> List[Image.Image]:
    images = []

    async with aiohttp.ClientSession(loop=loop) as session:
        responses = await asyncio.gather(*[asyncio.ensure_future(session.get(img_url)) for img_url in url_list])
        payloads = await asyncio.gather(*[asyncio.ensure_future(read_response(r)) for r in responses])

    for p in payloads:
        if p is None:
            images.append(None)
            continue

        try:
            images.append(Image.open(io.BytesIO(p)))
        except Exception as e:
            images.append(None)
            print(e)
    
    return images

def get_images_wrapper(url_list: List[str]) -> List[Image.Image]:
    loop = asyncio.get_event_loop()
    try:
        images = loop.run_until_complete(get_images_async(url_list, loop))
    except Exception as e:
        print(e)
        images = get_images_wrapper(url_list)
    return images

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
    img_hash_map: Dict[str, imagehash.ImageHash] = {k: None for k in img_id_list}
    
    for img_id_head, img_id_subset in img_id_grp.items():
        file_path = STORED_HASH_FOLDER + img_id_head + '.csv'
        if not os.path.exists(file_path):
            continue
        
        stored_df = pd.read_csv(file_path, names=['img_id', 'hash'])
        filtered_df = stored_df[stored_df['img_id'].isin(img_id_subset)]

        if len(filtered_df) == 0:
            continue
        
        filtered_df.apply(lambda x: img_hash_map.update({x['img_id']: imagehash.hex_to_hash(str(x['hash']))}), axis=1)

    return img_hash_map

def get_file_status(img_id_head: str) -> int:
    array_index = int(img_id_head, 16)
    return synchronizing_array[array_index]

def set_file_status(img_id_head: str, new_value: int) -> None:
    global synchronizing_array
    array_index = int(img_id_head, 16)
    synchronizing_array[array_index] = new_value

def acquire_file(img_id_head: str) -> None:
    while get_file_status(img_id_head) == 1:
        pass
    set_file_status(img_id_head, 1)

def release_file(img_id_head: str) -> None:
    set_file_status(img_id_head, 0)

def update_stored_hash(img_hash_map: Dict[str, imagehash.ImageHash]) -> None:
    img_hash_map = {k: v for k,v in img_hash_map.items() if (not pd.isna(k)) and (not pd.isna(v))}
    img_id_grp = get_img_id_grp(img_hash_map.keys())
    
    for img_id_head, img_id_subset in img_id_grp.items():
        file_path = STORED_HASH_FOLDER + img_id_head + '.csv'
        
        acquire_file(img_id_head)

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, names=['img_id', 'hash'])
        else:
            existing_df = None
        
        to_update_df = pd.DataFrame.from_dict(
            {k: v for k, v in img_hash_map.items() if k in img_id_subset}, 
            orient='index', 
            columns=['hash']
        )
        to_update_df = to_update_df.reset_index().rename(columns={'index': 'img_id'})
        to_update_df = to_update_df.dropna(subset=['hash'])

        final_df = pd.concat([existing_df, to_update_df])
        final_df = final_df.drop_duplicates()
        final_df.to_csv(file_path, index=False, header=False)

        release_file(img_id_head)

def img_hash_processing_function(img_id_list: List[str]) -> Dict[str, imagehash.ImageHash]:
    already_exist_hash_map = get_stored_hashes(img_id_list)

    image_ids_to_download = [k for k, v in already_exist_hash_map.items() if v is None]
    if not image_ids_to_download:
        return None

    new_img_urls = ['https://cf.shopee{}/file/{}'.format(DOMAIN_MAP.get(REGION), img_id) for img_id in image_ids_to_download]
    new_images = get_images_wrapper(new_img_urls)

    hash_list: List[imagehash.ImageHash] = []
    for i, img in enumerate(new_images):
        if isinstance(img, Image.Image):
            try:
                img_hash = imagehash.average_hash(img, 16)
            except Exception as e:
                print('Error occured when computing hash for', image_ids_to_download[i])
                print(e)
                hash_list.append(None)
            else:
                hash_list.append(img_hash)
        else:
            hash_list.append(None)
    
    img_hash_map = {k: v for k, v in zip(image_ids_to_download, hash_list)}
    update_stored_hash(img_hash_map)
    return img_hash_map

def batch_img_id_generator(file_name_list: Iterable[str], split_size: int, splits_per_yield: int) -> Generator[List[List[str]], None, None]:
    img_id_split_queue: Deque[List[str]] = deque()
    for file_name in file_name_list:
        print('Opening', file_name)
        df = pd.read_csv(TO_CHECK_FOLDER + file_name, names=['grass_region','item_id','img_id'])
        df= df[df['img_id'].map(lambda x: set(str(x).lower()).issubset(set('0123456789abcdef')))]
        img_id_list = df['img_id'].unique().astype(str)
        img_id_list.sort()
        img_id_list: List[str] = img_id_list.tolist()
        num_of_splits = len(img_id_list) // split_size
        img_id_splits = np.array_split(img_id_list, num_of_splits)
        img_id_split_queue.extend(img_id_splits)

        while len(img_id_split_queue) >= splits_per_yield:
            yield [img_id_split_queue.popleft() for _ in range(splits_per_yield)]
        else:
            yield [img_id_split_queue.popleft() for _ in range(len(img_id_split_queue))]

def main() -> None:
    print('Set up generator for image ID')
    files_to_check = os.listdir(TO_CHECK_FOLDER)
    files_to_check.sort()
    img_id_generator = batch_img_id_generator(files_to_check, SPLIT_SIZE, PARALLEL_PROCESSES)
    global synchronizing_array
    synchronizing_array = Array('B', 16**4)
    print('Start looping over image IDs')
    with Pool(processes=PARALLEL_PROCESSES) as pool:
        for i, img_id_list in enumerate(img_id_generator):
            img_hash_list = pool.map(img_hash_processing_function, img_id_list)
            if not any(img_hash_list):
                print('[{}] No new images in part {}'.format(datetime.datetime.now(), i+1))
                continue

            print('[{}] Part {} done'.format(datetime.datetime.now(), i+1))

if __name__ == '__main__':
    assert options.reg is not None
    assert options.folder is not None
    main()
    print('Done')