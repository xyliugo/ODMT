import os
from random import randint
from utils.dataset_config import DatasetConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = os.path.abspath(
    os.path.join(os.path.join(BASE_DIR, "../../.."), "home/public/lxy")
)
root_model_dir = root_data_dir

## dataset
dataset, tag = "KS", "10wu"
ds_config = DatasetConfig(dataset, tag)
(
    behaviors,
    text_data,
    image_data,
    text_model_load,
    max_seq_len,
) = ds_config.get_dataset_config()

## before training
item_tower = "modal"
image_model_load = "vit-b-32-clip"  # 'vit-b-32-clip' 'vit-base-mae'
gpu_id = "8,9"
master_port = randint(1, 1e5)

## training
seed = 1
lr = 1e-4
epoch = 40
version = "v1"
load_ckpt_name = "None"  # 'epoch-200.pt'
embedding_dim_list = [768]
id_embedding_dim, text_embedding_dim, image_embedding_dim = 768, 768, 768
tau_list, alpha_list = [0.5], [50]
merged_att_layers_list = [2]
init_list = [3]

label_screen = "ed{}_lr{}_len{}".format(embedding_dim_list[0], lr, max_seq_len)

for tau in tau_list:
    for alpha in alpha_list:
        for embedding_dim in embedding_dim_list:
            for merged_att_layers, init in zip(merged_att_layers_list, init_list):
                run_py = "CUDA_VISIBLE_DEVICES='{}' \
                                        /opt/anaconda3/envs/torch1.8/bin/python -m torch.distributed.launch \
                                        --nproc_per_node {} --master_port {} main.py \
                                        --root_data_dir {} --root_model_dir {} --dataset {} --behaviors {} --text_data {}  --image_data {}\
                                        --item_tower {} --load_ckpt_name {} --label_screen {} --max_seq_len {}\
                                        --lr {} --image_model_load {} --text_model_load {} --epoch {}\
                                        --embedding_dim {} --id_embedding_dim {} --text_embedding_dim {} --image_embedding_dim {}\
                                        --version {} --seed {} --merged_att_layers {} --init {} --alpha {} --tau {}".format(
                    gpu_id,
                    len(gpu_id.split(",")),
                    master_port,
                    root_data_dir,
                    root_model_dir,
                    dataset,
                    behaviors,
                    text_data,
                    image_data,
                    item_tower,
                    load_ckpt_name,
                    label_screen,
                    max_seq_len,
                    lr,
                    image_model_load,
                    text_model_load,
                    epoch,
                    embedding_dim,
                    id_embedding_dim,
                    text_embedding_dim,
                    image_embedding_dim,
                    version,
                    seed,
                    merged_att_layers,
                    init,
                    alpha,
                    tau,
                )

                os.system(run_py)
