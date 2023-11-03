import os
import math
import lmdb
import torch
import pickle
import random

import numpy as np
import torchvision as tv
import torch.distributed as dist

import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class LMDB_VIDEO:
    def __init__(self, video):
        self.video = video.tobytes()


# class LMDB_Image:
#     def __init__(self, image, id):
#         self.image = image.tobytes()


class LMDB_Image:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)


class ItemsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.item_num = len(data)
        self.data_id = np.arange(self.item_num)

    def __getitem__(self, idx):
        return torch.LongTensor(np.array(self.data_id[idx])), torch.LongTensor(
            np.array(self.data[idx])
        )

    def __len__(self):
        return self.data.shape[0]


class BuildMergedEvalDataset(Dataset):
    def __init__(self, data_text, data_image, item_id_to_keys, db_path, resize, args):
        self.data_image = data_image
        self.data_text = data_text
        self.item_id_to_keys = item_id_to_keys
        self.db_path = db_path
        self.resize = resize
        self.padding_emb = Image.fromarray(
            np.zeros((224, 224, 3)).astype("uint8")
        ).convert("RGB")
        self.args = args

        self.transform = transforms.Compose(
            [
                tv.transforms.Resize((self.resize, self.resize)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return self.data_image.shape[0]

    def __getitem__(self, index):
        # text
        text = self.data_text[index]
        # image
        item_id_image = self.data_image[index]
        if index == 0:
            return torch.LongTensor(text), self.transform(self.padding_emb)

        env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with env.begin() as txn:
            if self.args.behaviors in [
                "5wu1wi_ks_pairs.tsv",
                "Arts_Crafts_and_Sewing_user_behaviours.tsv",
                "Office_Products_user_behaviours.tsv",
            ]:
                byteflow = txn.get(self.item_id_to_keys[item_id_image].encode())
                IMAGE = pickle.loads(byteflow)
                img = self.transform(Image.fromarray(IMAGE.get_image()).convert("RGB"))
            elif self.args.behaviors in ["20wu4wi_ks_pairs.tsv", "10wu_ks_pairs.tsv"]:
                byteflow = txn.get(
                    "{}".format(self.item_id_to_keys[item_id_image]).encode("ascii")
                )
                IMAGE = pickle.loads(byteflow)
                img = torch.from_numpy(
                    np.frombuffer(IMAGE.image, dtype=np.float32).reshape(3, 224, 224)
                )
            elif self.args.behaviors in [
                "hm_pick_users_20W.tsv",
                "hm_pick_users_5W.tsv",
                "bili_pick_users_10W.tsv",
            ]:
                byteflow = txn.get(self.item_id_to_keys[item_id_image])
                IMAGE = pickle.loads(byteflow)
                img = self.transform(Image.fromarray(IMAGE.get_image()).convert("RGB"))

        return torch.LongTensor(text), torch.FloatTensor(img)


class ModalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, text_size):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len = max_seq_len + 1
        self.text_size = text_size

    def __len__(self):
        return len(self.u2seq)

    def worker_init_fn(self, worker_id):
        initial_seed = torch.initial_seed() % 2**31
        worker_seed = (
            initial_seed + worker_id + self.args.local_rank + 8 * self.args.node_rank
        )
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def __getitem__(self, index):
        seq = self.u2seq[index]
        seq_Len = len(seq)
        tokens = seq[:-1]
        tokens_Len = len(tokens)
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items_text = np.zeros((self.max_seq_len, self.text_size * 2))
        sample_items_id = [0] * mask_len_head + seq

        ##################################### Text #####################################
        for i in range(tokens_Len):
            # pos
            sample_items_text[mask_len_head + i] = self.item_content[seq[i]]
        # target
        sample_items_text[mask_len_head + tokens_Len] = self.item_content[seq[-1]]
        sample_items_text = torch.LongTensor(sample_items_text)
        sample_items_id = torch.LongTensor(sample_items_id)

        return sample_items_id, sample_items_text, torch.FloatTensor(log_mask)


class ImageDataset(Dataset):
    def __init__(
        self, u2seq, item_num, max_seq_len, db_path, item_id_to_keys, resize, args
    ):
        self.u2seq = u2seq
        self.args = args
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.db_path = db_path
        self.item_id_to_keys = item_id_to_keys
        self.resize = resize

        self.transform = transforms.Compose(
            [
                tv.transforms.Resize((self.resize, self.resize)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = len(seq) - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = np.zeros((self.max_seq_len, 3, self.resize, self.resize))
        sample_id_items = [0] * mask_len_head + seq

        env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with env.begin() as txn:
            if self.args.behaviors in [
                "5wu1wi_ks_pairs.tsv",
                "Arts_Crafts_and_Sewing_user_behaviours.tsv",
                "Office_Products_user_behaviours.tsv",
            ]:
                for i in range(tokens_Len):
                    # pos
                    IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[i]].encode()))
                    image_trans = self.transform(
                        Image.fromarray(IMAGE.get_image()).convert("RGB")
                    )
                    sample_items[mask_len_head + i] = image_trans
                # target
                IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
                image_trans = self.transform(
                    Image.fromarray(IMAGE.get_image()).convert("RGB")
                )
                sample_items[mask_len_head + tokens_Len] = image_trans
            elif self.args.behaviors in ["20wu4wi_ks_pairs.tsv", "10wu_ks_pairs.tsv"]:
                for i in range(tokens_Len):
                    # pos
                    IMAGE = pickle.loads(
                        txn.get(
                            "{}".format(self.item_id_to_keys[seq[i]]).encode("ascii")
                        )
                    )
                    image_trans = torch.from_numpy(
                        np.frombuffer(IMAGE.image, dtype=np.float32).reshape(
                            3, 224, 224
                        )
                    )
                    sample_items[mask_len_head + i] = image_trans
                # target
                IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
                image_trans = torch.from_numpy(
                    np.frombuffer(IMAGE.image, dtype=np.float32).reshape(3, 224, 224)
                )
                sample_items[mask_len_head + tokens_Len] = image_trans
            elif self.args.behaviors in [
                "hm_pick_users_20W.tsv",
                "hm_pick_users_5W.tsv",
                "bili_pick_users_10W.tsv",
            ]:
                for i in range(tokens_Len):
                    # pos
                    IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[i]]))
                    image_trans = self.transform(
                        Image.fromarray(IMAGE.get_image()).convert("RGB")
                    )
                    sample_items[mask_len_head + i] = image_trans
                # target
                IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]]))
                image_trans = self.transform(
                    Image.fromarray(IMAGE.get_image()).convert("RGB")
                )
                sample_items[mask_len_head + tokens_Len] = image_trans

        sample_id_items = torch.LongTensor(sample_id_items)
        sample_items = torch.FloatTensor(sample_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)


class TextDataset(Dataset):
    def __init__(self, userseq, item_content, max_seq_len, item_num, text_size):
        self.userseq = userseq
        self.item_content = item_content
        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num
        self.text_size = text_size

    def __len__(self):
        return len(self.userseq)

    def __getitem__(self, index):
        seq = self.userseq[index]
        seq_Len = len(seq)
        tokens = seq[:-1]
        tokens_Len = len(tokens)
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_id_items = [0] * mask_len_head + seq
        sample_items = np.zeros((self.max_seq_len, self.text_size * 2))
        for i in range(tokens_Len):
            # pos
            sample_items[mask_len_head + i] = self.item_content[seq[i]]
        # target
        sample_items[mask_len_head + tokens_Len] = self.item_content[seq[-1]]
        sample_items = torch.FloatTensor(sample_items)
        sample_id_items = torch.LongTensor(sample_id_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)


class VideoDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, item_id_to_keys, db_path):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.item_id_to_keys = item_id_to_keys
        self.video_lmdb_path = db_path

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = len(seq) - 1
        mask_len = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len + [1] * tokens_Len

        sample_items = np.zeros((self.max_seq_len, 4, 3, 224, 224))
        sample_id_items = [0] * mask_len + seq

        env = lmdb.open(
            self.video_lmdb_path,
            subdir=os.path.isdir(self.video_lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        neg_items = []
        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                vdo = pickle.loads(
                    txn.get("{}".format(self.item_id_to_keys[seq[i]]).encode("ascii"))
                )
                vdo = np.copy(np.frombuffer(vdo.video, dtype=np.float32)).reshape(
                    4, 3, 224, 224
                )
                sample_items[mask_len + i] = vdo

            # target
            vdo = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
            vdo = np.copy(np.frombuffer(vdo.video, dtype=np.float32)).reshape(
                4, 3, 224, 224
            )
            sample_items[0][mask_len + tokens_Len] = vdo

        sample_id_items = torch.LongTensor(sample_id_items)
        sample_items = torch.FloatTensor(sample_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)


class IdDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, args):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.args = args

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = [0] * mask_len_head + seq
        sample_items = torch.LongTensor(np.array(sample_items))

        return sample_items, torch.FloatTensor(log_mask)


class IdEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class EvalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num):
        self.u2seq = u2seq
        self.item_embs = item_content[0]
        self.text_embs = item_content[1]
        self.image_embs = item_content[2]
        self.max_seq_len = max_seq_len
        self.item_num = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        item_embs = self.item_embs[pad_tokens]
        text_embs = self.text_embs[pad_tokens]
        image_embs = self.image_embs[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return (
            torch.LongTensor([user_id]),
            item_embs,
            text_embs,
            image_embs,
            torch.FloatTensor(log_mask),
            labels,
        )


class LmdbEvalDataset(Dataset):
    def __init__(self, data, item_id_to_keys, db_path, resize, mode, args):
        self.data = data
        self.item_id_to_keys = item_id_to_keys
        self.db_path = db_path
        self.resize = resize
        self.mode = mode
        self.args = args
        if mode == "image":
            self.padding_emb = Image.fromarray(
                np.zeros((224, 224, 3)).astype("uint8")
            ).convert("RGB")
        else:
            self.padding_emb = torch.zeros((4, 3, 224, 224))

        self.transform = transforms.Compose(
            [
                tv.transforms.Resize((self.resize, self.resize)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        item_id = self.data[index]
        if index == 0:
            if self.mode == "image":
                return self.transform(self.padding_emb)
            else:
                return torch.zeros((4, 3, 224, 224))

        env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin() as txn:
            if self.args.behaviors in [
                "5wu1wi_ks_pairs.tsv",
                "Arts_Crafts_and_Sewing_user_behaviours.tsv",
                "Office_Products_user_behaviours.tsv",
            ]:
                byteflow = txn.get(self.item_id_to_keys[item_id].encode())
                IMAGE = pickle.loads(byteflow)
                output = self.transform(
                    Image.fromarray(IMAGE.get_image()).convert("RGB")
                )
            elif self.args.behaviors in ["20wu4wi_ks_pairs.tsv", "10wu_ks_pairs.tsv"]:
                byteflow = txn.get(
                    "{}".format(self.item_id_to_keys[item_id]).encode("ascii")
                )
                IMAGE = pickle.loads(byteflow)
                output = torch.from_numpy(
                    np.frombuffer(IMAGE.image, dtype=np.float32).reshape(3, 224, 224)
                )
            elif self.args.behaviors in [
                "hm_pick_users_20W.tsv",
                "hm_pick_users_5W.tsv",
                "bili_pick_users_10W.tsv",
            ]:
                byteflow = txn.get(self.item_id_to_keys[item_id])
                IMAGE = pickle.loads(byteflow)
                output = self.transform(
                    Image.fromarray(IMAGE.get_image()).convert("RGB")
                )

        return torch.FloatTensor(output)


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = (
            int(
                math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)
            )
            * self.batch_size
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        return iter(indices)

    def __len__(self):
        return self.num_samples
