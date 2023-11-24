
import os
import lmdb
import pickle
import tqdm
import torch
import argparse
import numpy as np
import pandas as pd

from PIL import Image

torch.manual_seed(123456)

class LMDB_Image:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Arts_Crafts_and_Sewing')
    parser.add_argument("--resize", type=int, default=224)

    args = parser.parse_args()
    dataset = args.dataset
    resize = args.resize

    with open(f'{dataset}_valid_item_id.pkl', 'rb') as file:
        valid_item_id_list = pickle.load(file)
    image_num = len(valid_item_id_list)

    lmdb_path = f'{dataset}_image.lmdb'
    isdir = os.path.isdir(lmdb_path)
    lmdb_env = lmdb.open(lmdb_path, subdir=isdir, map_size=image_num * np.zeros((3, resize, resize)).nbytes * 10,
                         readonly=False, meminit=False, map_async=True)
    txn = lmdb_env.begin(write=True)
    
    image_file = f'images'
    bad_file = {}
    lmdb_keys = []
    write_frequency = 5000
    for index, row in tqdm.tqdm(enumerate(valid_item_id_list)):
        item_id = str(row)
        item_name = row + '.jpg'
        lmdb_keys.append(item_id)
        try:
            img = np.array(Image.open(os.path.join(image_file, item_name)).convert('RGB'))
            temp = LMDB_Image(img, item_id)
            txn.put(u'{}'.format(item_id).encode('ascii'), pickle.dumps(temp))
            if index % write_frequency == 0 and index != 0:
                txn.commit()
                txn = lmdb_env.begin(write=True)
        except Exception as e:
            bad_file[index] = item_id

    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in lmdb_keys]
    with lmdb_env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', pickle.dumps(len(keys)))
    print(len(keys))
    print("Flushing database ...")
    lmdb_env.sync()
    lmdb_env.close()
    print('bad_file  ', len(bad_file))
    for k, v in bad_file.items():
        print(k, v)

