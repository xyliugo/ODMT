class DatasetConfig:
    def __init__(self, dataset, tag=None):
        if dataset == "KS" and tag == "20wu4wi":
            self.behaviors = tag + "_ks_pairs.tsv"
            self.text_data = tag + "_ks_title.csv"
            self.image_data = tag + "_ks_images.lmdb"
            self.text_model_load = "chinese-roberta-wwm-ext"  # 'bert-base-cn'
            self.max_seq_len = 12

        elif dataset == "KS" and tag == "10wu":
            self.behaviors = tag + "_ks_pairs.tsv"
            self.text_data = tag + "_ks_title.csv"
            self.image_data = tag + "_ks_cover.lmdb"
            self.text_model_load = "chinese-roberta-wwm-ext"  # 'bert-base-cn'
            self.max_seq_len = 12

        elif dataset == "hm" and tag == "5W":
            self.behaviors = "hm_pick_users_5W.tsv"
            self.text_data = "hm_pick_items_5W.tsv"
            self.image_data = "hm_image_20W.lmdb"
            self.text_model_load = "bert-base-uncased"  # 'bert-base-cn'
            self.max_seq_len = 12

        elif dataset == "bili":
            self.behaviors = "bili_pick_users_10W.tsv"
            self.text_data = "bili_pick_items_10W.tsv"
            self.image_data = "bili_image_10W.lmdb"
            self.text_model_load = "chinese-roberta-wwm-ext"  # 'bert-base-cn'
            self.max_seq_len = 15

        elif dataset == "amazon":
            self.behaviors = f"{tag}_user_behaviours.tsv"
            self.text_data = f"{tag}_text.tsv"
            self.image_data = f"{tag}_image.lmdb"
            self.text_model_load = "bert-base-uncased"  # 'bert-base-cn'
            self.max_seq_len = 12

    def get_dataset_config(self):
        return (
            self.behaviors,
            self.text_data,
            self.image_data,
            self.text_model_load,
            self.max_seq_len,
        )
