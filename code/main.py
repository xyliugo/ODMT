from warnings import simplefilter
from transformers import logging

logging.set_verbosity_warning()
simplefilter(action="ignore", category=UserWarning)
simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=DeprecationWarning)

import os
import re
import time
import torch
import random

import numpy as np
import torch.optim as optim
import torch.distributed as dist

from torch import nn
from pathlib import Path
from statistics import mode
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast as autocast
from transformers import CLIPVisionModel, SwinForImageClassification, ViTMAEModel
from transformers import (
    BertModel,
    BertTokenizer,
    BertConfig,
    RobertaTokenizer,
    RobertaModel,
    RobertaConfig,
)

from model.model import Model
from model.feature_extractors import extract_text_feature, extract_image_feature
from utils.parameters import parse_args
from utils.load_data import (
    read_texts,
    read_behaviors_text,
    read_items,
    read_behaviors,
    get_doc_input_bert,
)
from utils.logging_utils import (
    para_and_log,
    report_time_train,
    report_time_eval,
    save_model,
    setuplogger,
    get_time,
)
from utils.dataset import ModalDataset, LMDB_Image
from utils.metrics import eval_model, get_merged_fusion_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
scaler = torch.cuda.amp.GradScaler()


def setup_seed(seed):
    """
    global seed config
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train(args, model_dir, Log_file, Log_screen, start_time, local_rank):
    # ========================================== Text and Image Encoders ===========================================
    if "roberta-base-en" in args.text_model_load:
        Log_file.info("load roberta model...")
        text_model_load = os.path.abspath(
            os.path.join(
                args.root_model_dir, "pretrained_models/bert", args.text_model_load
            )
        )
        tokenizer = RobertaTokenizer.from_pretrained(text_model_load)
        config = RobertaConfig.from_pretrained(
            text_model_load, output_hidden_states=True
        )
        text_model = RobertaModel.from_pretrained(text_model_load, config=config)

    elif "chinese-roberta-wwm-ext" in args.text_model_load:
        Log_file.info("load Chinese Roberta model...")
        text_model_load = os.path.abspath(
            os.path.join(
                args.root_model_dir, "pretrained_models/bert", args.text_model_load
            )
        )
        tokenizer = BertTokenizer.from_pretrained(text_model_load)
        text_model = BertModel.from_pretrained(text_model_load)

    else:
        Log_file.info("load bert model...")
        text_model_load = os.path.abspath(
            os.path.join(
                args.root_model_dir, "pretrained_models/bert", args.text_model_load
            )
        )
        tokenizer = BertTokenizer.from_pretrained(text_model_load)
        config = BertConfig.from_pretrained(text_model_load, output_hidden_states=True)
        text_model = BertModel.from_pretrained(text_model_load, config=config)

    if "vit-b-32-clip" in args.image_model_load:
        Log_file.info("load Vit model...")
        image_model_load = os.path.abspath(
            os.path.join(
                args.root_model_dir, "pretrained_models", args.image_model_load
            )
        )
        image_model = CLIPVisionModel.from_pretrained(image_model_load)

    elif "vit-base-mae" in args.image_model_load:
        Log_file.info("load MAE model...")
        image_model_load = os.path.abspath(
            os.path.join(
                args.root_model_dir, "pretrained_models", args.image_model_load
            )
        )
        image_model = ViTMAEModel.from_pretrained(image_model_load)

    elif "swin_base" in args.image_model_load:
        image_model_load = os.path.abspath(
            os.path.join(
                args.root_model_dir, "pretrained_models", args.image_model_load
            )
        )
        image_model = SwinForImageClassification.from_pretrained(image_model_load)

    elif "swin_tiny" in args.image_model_load:
        image_model_load = os.path.abspath(
            os.path.join(
                args.root_model_dir, "pretrained_models", args.image_model_load
            )
        )
        image_model = SwinForImageClassification.from_pretrained(image_model_load)

    # ========================================== Loading Data ===========================================
    item_content = None
    item_id_to_keys = None

    Log_file.info("read texts ...")
    (
        item_dic_titles_after_tokenizer,
        before_item_name_to_index,
        before_item_index_to_name,
    ) = read_texts(tokenizer, args)

    Log_file.info("read behaviors ...")
    (
        item_num,
        item_dic_titles_after_tokenizer,
        item_name_to_index,
        users_train,
        users_valid,
        users_history_for_valid,
        users_test,
        users_history_for_test,
        pop_prob_list,
    ) = read_behaviors_text(
        item_dic_titles_after_tokenizer,
        before_item_name_to_index,
        before_item_index_to_name,
        Log_file,
        args,
    )

    Log_file.info("combine text information...")
    text_title, text_title_attmask = get_doc_input_bert(
        item_dic_titles_after_tokenizer, item_name_to_index, args
    )

    item_content = np.concatenate([text_title, text_title_attmask], axis=1)

    Log_file.info("read images/videos/id...")
    before_item_id_to_keys, before_item_name_to_id = read_items(args)

    Log_file.info("read behaviors...")
    (
        item_num,
        item_id_to_keys,
        users_train,
        users_valid,
        users_history_for_valid,
        users_test,
        users_history_for_test,
        pop_prob_list,
    ) = read_behaviors(before_item_id_to_keys, before_item_name_to_id, Log_file, args)

    t_feat = extract_text_feature(text_model, item_content, local_rank, args)
    v_feat = extract_image_feature(
        image_model, item_num, item_id_to_keys, local_rank, args
    )

    # ========================================== Building Model ===========================================
    Log_file.info("build model...")
    model = Model(args, pop_prob_list, item_num, t_feat, v_feat).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    if "epoch" in args.load_ckpt_name:
        Log_file.info("load ckpt if not None...")
        #############
        ckpt_path = os.path.abspath(os.path.join(model_dir, args.load_ckpt_name))
        start_epoch = int(re.split(r"[._-]", args.load_ckpt_name)[1])
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        Log_file.info("load checkpoint...")
        model.load_state_dict(checkpoint["model_state_dict"])
        Log_file.info(f"Model loaded from {args.load_ckpt_name}")
        torch.set_rng_state(checkpoint["rng_state"])  # load torch的随机数生成器状态
        torch.cuda.set_rng_state(
            checkpoint["cuda_rng_state"]
        )  # load torch.cuda的随机数生成器状态
        is_early_stop = False
    else:
        checkpoint = None  # new
        ckpt_path = None  # new
        start_epoch = 0
        is_early_stop = False

    Log_file.info("model.cuda()...")
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # ============================ Dataset and Dataloader ============================
    Log_file.info("build  text and image dataset...")
    train_dataset = ModalDataset(
        u2seq=users_train,
        item_content=item_content,
        max_seq_len=args.max_seq_len,
        text_size=args.num_words_title,
    )

    Log_file.info("build DDP sampler...")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2**31
        worker_seed = initial_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    Log_file.info("build dataloader...")
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        multiprocessing_context="fork",
        worker_init_fn=worker_init_reset_seed,
        pin_memory=True,
        sampler=train_sampler,
    )

    # ============================ Optimizer ============================
    params = []

    for index, (name, param) in enumerate(model.module.named_parameters()):
        if param.requires_grad:
            params.append(param)

    optimizer = optim.AdamW(
        [
            {
                "params": params,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "initial_lr": args.lr,
            }
        ]
    )

    for children_model in optimizer.state_dict()["param_groups"]:
        Log_file.info(
            "***** {} parameters have learning rate {} *****".format(
                len(children_model["params"]), children_model["lr"]
            )
        )

    Log_file.info("***** {} parameters with grad *****".format(len(list(params))))

    if "None" not in args.load_ckpt_name:  # load 优化器状态
        optimizer.load_state_dict(checkpoint["optimizer"])
        Log_file.info(f"optimizer loaded from {ckpt_path}")

    # ============================  training  ============================
    Log_file.info("\n")
    Log_file.info("Training...")
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0
    steps_for_log, steps_for_eval = para_and_log(
        model,
        len(users_train),
        args.batch_size,
        Log_file,
        logging_num=args.logging_num,
        testing_num=args.testing_num,
    )
    Log_screen.info("{} train start".format(args.label_screen))
    epoch_left = args.epoch - start_epoch

    for ep in range(epoch_left):
        now_epoch = start_epoch + ep + 1
        train_dl.sampler.set_epoch(now_epoch)
        loss, batch_index, need_break = 0.0, 1, False
        align, uniform = 0.0, 0.0
        ce_loss, kl_loss = 0.0, 0.0

        if not need_break and (now_epoch - 1) % 1 == 0 and now_epoch > 1:
            (
                max_eval_value,
                max_epoch,
                early_stop_epoch,
                early_stop_count,
                need_break,
            ) = eval(
                now_epoch,
                max_epoch,
                early_stop_epoch,
                max_eval_value,
                early_stop_count,
                model,
                users_history_for_valid,
                users_valid,
                128,
                item_num,
                "train",
                is_early_stop,
                local_rank,
                args,
                Log_file,
                pop_prob_list,
                now_epoch,
                item_content,
                item_id_to_keys,
            )

        if not need_break and (now_epoch - 1) % 1 == 0 and now_epoch > 1:
            _, _, _, _, _ = eval(
                now_epoch,
                max_epoch,
                early_stop_epoch,
                max_eval_value,
                early_stop_count,
                model,
                users_history_for_test,
                users_test,
                128,
                item_num,
                "test",
                is_early_stop,
                local_rank,
                args,
                Log_file,
                pop_prob_list,
                now_epoch,
                item_content,
                item_id_to_keys,
            )

        Log_file.info("\n")
        Log_file.info("epoch {} start".format(now_epoch))
        Log_file.info("")

        model.train()

        for data in train_dl:
            sample_items_id, sample_items_text, log_mask = data
            sample_items_id, sample_items_text, log_mask = (
                sample_items_id.to(local_rank),
                sample_items_text.to(local_rank),
                log_mask.to(local_rank),
            )
            sample_items_text = sample_items_text.view(-1, args.num_words_title * 2)
            sample_items_id = sample_items_id.view(-1)

            optimizer.zero_grad()

            # # 混合精度（加速）
            with autocast(enabled=True):
                bz_ce_loss, bz_kl_loss = model(
                    now_epoch, sample_items_id, sample_items_text, log_mask, local_rank
                )
                ce_loss += bz_ce_loss.data.float()
                kl_loss += bz_kl_loss.data.float()

            scaler.scale(bz_ce_loss + bz_kl_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.module.parameters(), max_norm=5, norm_type=2
            )
            scaler.step(optimizer)
            scaler.update()

            if torch.isnan(ce_loss.data):
                need_break = True
                break

            if batch_index % steps_for_log == 0:
                Log_file.info(
                    "Ed: {}, ce loss: {:.3f}, kl loss: {:.3f}, align: {:.3f}, uniform: {:.3f}".format(
                        batch_index * args.batch_size,
                        ce_loss / batch_index,
                        kl_loss / batch_index,
                        align / batch_index,
                        uniform / batch_index,
                    )
                )
            batch_index += 1

        if dist.get_rank() == 0 and now_epoch % args.save_step == 0:
            save_model(
                now_epoch,
                model,
                model_dir,
                optimizer,
                torch.get_rng_state(),
                torch.cuda.get_rng_state(),
                Log_file,
            )  # new

        Log_file.info("")
        next_set_start_time = report_time_train(
            batch_index,
            now_epoch,
            ce_loss,
            align / batch_index,
            uniform / batch_index,
            next_set_start_time,
            start_time,
            Log_file,
        )
        Log_screen.info(
            "{} training: epoch {}/{}".format(args.label_screen, now_epoch, args.epoch)
        )

        if need_break:
            break

    Log_file.info("\n")
    Log_file.info("%" * 90)
    Log_file.info(
        "max eval Hit10 {:0.5f}  in epoch {}".format(max_eval_value * 100, max_epoch)
    )
    Log_file.info("early stop in epoch {}".format(early_stop_epoch))
    Log_file.info("the End")
    Log_screen.info(
        "{} train end in epoch {}".format(args.label_screen, early_stop_epoch)
    )


def eval(
    now_epoch,
    max_epoch,
    early_stop_epoch,
    max_eval_value,
    early_stop_count,
    model,
    user_history,
    users_eval,
    batch_size,
    item_num,
    mode,
    is_early_stop,
    local_rank,
    args,
    Log_file,
    pop_prob_list,
    epoch,
    item_content=None,
    item_id_to_keys=None,
):
    eval_start_time = time.time()
    Log_file.info("Validating based on {}".format(args.item_tower))
    item_scoring = get_merged_fusion_score(
        model, item_content, item_num, batch_size, args, local_rank
    )
    valid_Hit10, nDCG10 = eval_model(
        model,
        user_history,
        users_eval,
        item_scoring,
        batch_size,
        args,
        item_num,
        Log_file,
        mode,
        pop_prob_list,
        epoch,
        local_rank,
    )

    report_time_eval(eval_start_time, Log_file)
    Log_file.info("")
    need_break = False

    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        max_epoch = now_epoch
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count > 5:
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch

    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break


def main():
    args = parse_args()

    # ============== Distributed Computation Config ==============
    local_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # ============== Experiment and Logging Config ===============
    setup_seed(args.seed + dist.get_rank())  # magic number

    assert args.item_tower in ["modal", "text", "image", "id"]
    dir_label = str(args.behaviors).strip().split(".")[0] + "_" + str(args.item_tower)
    log_paras = (
        f"{args.seed}_init_{args.init}_{args.version}_tau_{args.tau}_alpha_{args.alpha}"
        f"_m_{args.merged_att_layers}"
        f"_bs_{args.batch_size}_ed_{args.embedding_dim}_{args.id_embedding_dim}"
        f"_lr_{args.lr}_maxLen_{args.max_seq_len}"
    )

    model_dir = os.path.join(
        "./checkpoint/checkpoint_" + dir_label, f"cpt_" + log_paras
    )
    time_run = ""  # avoid redundant log records
    Log_file, Log_screen = setuplogger(
        dir_label, log_paras, time_run, args.mode, dist.get_rank()
    )
    Log_file.info(args)

    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    train(args, model_dir, Log_file, Log_screen, start_time, local_rank)

    end_time = time.time()
    hour, minute, seconds = get_time(start_time, end_time)
    Log_file.info(
        "#### (time) all: hours {} minutes {} seconds {} ####".format(
            hour, minute, seconds
        )
    )


if __name__ == "__main__":
    main()
