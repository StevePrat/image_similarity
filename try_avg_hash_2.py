import asyncio
import enum
from PIL import Image
import imagehash
import requests
import io
import itertools
import aiohttp
from typing import *
from tqdm import tqdm
from multiprocessing.pool import Pool
import datetime
import optparse
import os
import pandas as pd
import numpy as np

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

async def read_response(response, extra_info: Any = None):
    try:
        if response.content_type.startswith('image'):
            return await response.read()
        else:
            return None
    except Exception as e:
        print(response)
        print(extra_info)
        raise e

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
    images = loop.run_until_complete(get_images_async(url_list, loop))
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
    img_hash_map = {k: None for k in img_id_list}
    
    for img_id_head, img_id_subset in img_id_grp.items():
        file_path = STORED_HASH_FOLDER + img_id_head + '.csv'
        if not os.path.exists(file_path):
            continue
        
        stored_df = pd.read_csv(file_path, names=['img_id', 'hash'])
        filtered_df = stored_df[stored_df['img_id'].isin(img_id_subset)]

        if len(filtered_df) == 0:
            continue

        img_hashes_to_add = filtered_df.apply(lambda x: {x['img_id']: imagehash.hex_to_hash(x['hash'])}, axis=1)
        for d in img_hashes_to_add:
            img_hash_map.update(d)

    return img_hash_map

def img_hash_processing_function(img_id_list: List[str]) -> Dict[str, imagehash.ImageHash]:
    already_exist_hash_map = get_stored_hashes(img_id_list)

    images_to_download = [k for k, v in already_exist_hash_map.items() if v is None]
    if not images_to_download:
        return None

    new_img_urls = ['https://cf.shopee{}/file/{}'.format(DOMAIN_MAP.get(REGION), img_id) for img_id in images_to_download]
    new_images = get_images_wrapper(new_img_urls)

    hash_list: List[imagehash.ImageHash] = []
    for i, img in enumerate(new_images):
        if isinstance(img, Image.Image):
            try:
                img_hash = imagehash.average_hash(img, 16)
            except Exception as e:
                print('Error occured when computing hash for', images_to_download[i])
                print(e)
                hash_list.append(None)
            else:
                hash_list.append(img_hash)
        else:
            hash_list.append(None)
    
    result = {k: v for k, v in zip(images_to_download, hash_list)}
    return result

def update_stored_hash(img_hash_map: Dict[str, imagehash.ImageHash]) -> None:
    img_id_grp = get_img_id_grp(img_hash_map.keys())
    
    for img_id_head, img_id_subset in img_id_grp.items():
        file_path = STORED_HASH_FOLDER + img_id_head + '.csv'
        
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

        final_df = pd.concat([existing_df, to_update_df])
        final_df = final_df.drop_duplicates()

        final_df.to_csv(file_path, index=False, header=False)

def batch_img_id_generator(file_name_list: Iterable[str], split_size: int) -> List[str]:
    for file_name in file_name_list:
        df = pd.read_csv(TO_CHECK_FOLDER + file_name, names=['grass_region','item_id','img_id'])
        img_id_list = df['img_id'].unique().tolist()
        num_of_splits = len(img_id_list) // split_size
        img_id_list_splitted = np.array_split(img_id_list, num_of_splits)

        for img_id_sublist in img_id_list_splitted:
            yield img_id_sublist

def main() -> None:
    print('Set up generator for image ID')
    files_to_check = os.listdir(TO_CHECK_FOLDER)
    files_to_check.sort()
    img_id_generator = batch_img_id_generator(files_to_check, SPLIT_SIZE)
    print('Start looping over image IDs')
    with Pool(processes=PARALLEL_PROCESSES) as pool:
        for i, img_hash_map in enumerate(pool.imap(img_hash_processing_function, img_id_generator)):
            img_hash_map: Dict[str, imagehash.ImageHash]
            if img_hash_map is None:
                print('No new images in this part')
                print('[{}] Part {} done'.format(datetime.datetime.now(), i+1))
                continue

            img_hash_map = {k: v for k, v in img_hash_map.items() if v is not None}
            update_stored_hash(img_hash_map)
            print('[{}] Part {} done'.format(datetime.datetime.now(), i+1))

if __name__ == '__main__':
    assert options.reg is not None
    assert options.folder is not None
    main()
    print('Done')