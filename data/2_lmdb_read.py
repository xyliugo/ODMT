import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import lmdb
import pickle
from PIL import Image
import os
import numpy as np
import tqdm
import argparse


class LMDB_Image:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)


class DatasetLMDB(Dataset):
    def __init__(self, db_path, resize):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transforms.Compose([
                tv.transforms.Resize((resize, resize)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __getitem__(self, index):
        env = self.env
        with env.begin() as txn:
            byteflow = txn.get(self.keys[index])
        IMAGE = pickle.loads(byteflow)
        img, id = IMAGE.get_image(), IMAGE.id
        return self.transform(Image.fromarray(img).convert('RGB')), id

    def __len__(self):
        return self.length


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Arts_Crafts_and_Sewing')
    parser.add_argument("--resize", type=int, default=224)
    args = parser.parse_args()

    dataset = args.dataset
    resize = args.resize

    print('read lmdb database')
    lmdb_data = DatasetLMDB(f'{dataset}_image.lmdb', resize=resize)
    print('all images', lmdb_data.length)
    lmdb_dl = DataLoader(lmdb_data, batch_size=512, num_workers=0, pin_memory=True)
    toPIL = transforms.ToPILImage()
    for data in tqdm.tqdm(lmdb_dl):
        images, ids = data
        for image, id in zip(images, ids):
            toPIL(image).show()
            print(type(image))
            print(image.shape)
            print(np.array(image).shape)
            print(type(id))
            print(id)
            break
