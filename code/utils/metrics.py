import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import os
import math
from .dataset import (
    EvalDataset,
    SequentialDistributedSampler,
    LmdbEvalDataset,
    IdEvalDataset,
    ItemsDataset,
    BuildMergedEvalDataset,
)

"""
    一些工具代码, 用于本py文件
"""


def alignment(x, y):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(2).mean()


def uniformity(x):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


def item_collate_fn(arr):
    arr = torch.LongTensor(np.array(arr))
    return arr


def id_collate_fn(arr):
    arr = torch.LongTensor(arr)
    return arr


def print_metrics(x, Log_file, v_or_t, mode):
    Log_file.info(
        mode
        + " "
        + v_or_t
        + "_results   {}".format("\t".join(["{:0.5f}".format(i * 100) for i in x]))
    )


def get_mean(arr):
    return [i.mean() for i in arr]


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


def eval_concat(eval_list, test_sampler):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = (
            distributed_concat(eval_m, len(test_sampler.dataset))
            .to(torch.device("cpu"))
            .numpy()
        )
        eval_result.append(eval_m_cpu.mean())
    return eval_result


def scoring_concat(scoring, test_sampler):
    scoring = distributed_concat(scoring, len(test_sampler.dataset))
    return scoring


def metrics_topK(y_score, y_true, item_rank, topK, local_rank):
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)
    rank = torch.sum(y_true * item_rank)
    eval_ra = torch.zeros(2).to(local_rank)
    if rank <= topK:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    return rank, eval_ra


def get_item_id_score(model, item_num, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = IdEvalDataset(data=np.arange(item_num + 1))
    item_dataloader = DataLoader(
        item_dataset,
        batch_size=test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=item_collate_fn,
    )
    item_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb = (
                model.module.id_encoder(input_ids).to(torch.device("cpu")).detach()
            )
            item_scoring.extend(item_emb)
    return torch.stack(tensors=item_scoring, dim=0)


def get_item_t_score(model, item_content, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = IdEvalDataset(data=np.arange(len(item_content)))
    item_dataloader = DataLoader(
        item_dataset,
        batch_size=test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    item_scoring = []
    with torch.no_grad():
        for input_id in item_dataloader:
            input_text = item_content[input_id]
            input_id = torch.LongTensor(input_id)
            input_text = torch.LongTensor(input_text)
            input_id, input_text = input_id.to(local_rank), input_text.to(local_rank)
            item_emb = model.module.text_encoder(input_id, input_text)
            item_scoring.extend(item_emb)

    return torch.stack(tensors=item_scoring, dim=0).to(torch.device("cpu")).detach()


def get_item_v_score(
    model, item_num, item_id_to_keys, test_batch_size, args, local_rank
):
    model.eval()
    item_dataset = LmdbEvalDataset(
        data=np.arange(item_num + 1),
        item_id_to_keys=item_id_to_keys,
        db_path=os.path.join(args.root_data_dir, args.dataset, args.image_data),
        resize=args.image_resize,
        mode="image",
        args=args,
    )
    item_dataloader = DataLoader(
        item_dataset,
        batch_size=test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    item_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb = model.module.image_encoder(input_ids)
            item_scoring.extend(item_emb)
    return torch.stack(tensors=item_scoring, dim=0).to(torch.device("cpu")).detach()


def get_fusion_score(model, item_scoring_text, item_scoring_image, local_rank, args):
    model.eval()
    with torch.no_grad():
        if "modal" in args.item_tower:
            item_scoring_text = item_scoring_text.to(local_rank)
            item_scoring_image = item_scoring_image.to(local_rank)
            item_scoring = model.module.fusion_module(
                item_scoring_text, item_scoring_image
            )

    return item_scoring.to(torch.device("cpu")).detach()


def get_merged_fusion_score(
    model, item_content, item_num, test_batch_size, args, local_rank
):
    model.eval()
    item_dataset = IdEvalDataset(data=np.arange(len(item_content)))
    item_dataloader = DataLoader(
        item_dataset,
        batch_size=test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    item_scoring, text_scoring, image_scoring = [], [], []
    with torch.no_grad():
        for input_id in item_dataloader:
            input_text = item_content[input_id]
            input_id, input_text = torch.LongTensor(input_id), torch.LongTensor(
                input_text
            )
            input_id, input_text = input_id.to(local_rank), input_text.to(local_rank)

            image_mask = torch.ones((input_text.size(0), 50)).to(local_rank)
            text_mask = torch.narrow(
                input_text, 1, args.num_words_title, args.num_words_title
            )

            id_embs = model.module.id_proj(model.module.id_encoder(input_id))
            t_embs = model.module.text_proj(model.module.t_feat[input_id])
            v_embs = model.module.image_proj(model.module.v_feat[input_id])
            id_embs, t_embs, v_embs = model.module.IMT(
                id_embs,
                t_embs,
                text_mask,
                v_embs,
                image_mask,
                model.module.args.version,
            )
            id_embs, t_embs, v_embs = (
                model.module.id_dnns(id_embs),
                model.module.t_dnns(t_embs),
                model.module.v_dnns(v_embs),
            )

            item_scoring.extend(id_embs)
            text_scoring.extend(t_embs)
            image_scoring.extend(v_embs)

    item_scorings = (
        torch.stack(tensors=item_scoring, dim=0).to(torch.device("cpu")).detach()
    )
    text_scorings = (
        torch.stack(tensors=text_scoring, dim=0).to(torch.device("cpu")).detach()
    )
    image_scorings = (
        torch.stack(tensors=image_scoring, dim=0).to(torch.device("cpu")).detach()
    )

    return item_scorings, text_scorings, image_scorings


def eval_model(
    model,
    user_history,
    eval_seq,
    item_scoring,
    test_batch_size,
    args,
    item_num,
    Log_file,
    v_or_t,
    pop_prob_list,
    epoch,
    local_rank,
):
    eval_dataset = EvalDataset(
        u2seq=eval_seq,
        item_content=item_scoring,
        max_seq_len=args.max_seq_len + 1,
        item_num=item_num,
    )
    test_sampler = SequentialDistributedSampler(
        eval_dataset, batch_size=test_batch_size
    )
    eval_dl = DataLoader(
        eval_dataset,
        batch_size=test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )
    model.eval()
    topK = 10
    Log_file.info(
        v_or_t
        + "_methods   {}".format(
            "\t".join(["Hit{}".format(topK), "nDCG{}".format(topK)])
        )
    )
    id_embs = item_scoring[0].to(local_rank)
    t_embs = item_scoring[1].to(local_rank)
    v_embs = item_scoring[2].to(local_rank)
    with torch.no_grad():
        eval_all, eval_text, eval_image, eval_id = [], [], [], []
        user_list, item_list, rank_list = [], [], []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_item, input_text, input_image, log_mask, labels = data
            user_ids, input_item, input_text, input_image, log_mask, labels = (
                user_ids.to(local_rank),
                input_item.to(local_rank),
                input_text.to(local_rank),
                input_image.to(local_rank),
                log_mask.to(local_rank),
                labels.to(local_rank).detach(),
            )

            id_user_emb = model.module.id_user_encoder(
                input_item, log_mask, local_rank
            )[:, -1].detach()
            t_user_emb = model.module.t_user_encoder(input_text, log_mask, local_rank)[
                :, -1
            ].detach()
            v_user_emb = model.module.v_user_encoder(input_image, log_mask, local_rank)[
                :, -1
            ].detach()
            id_scores = torch.matmul(id_user_emb, id_embs.t()).squeeze(dim=-1).detach()
            t_scores = torch.matmul(t_user_emb, t_embs.t()).squeeze(dim=-1).detach()
            v_scores = torch.matmul(v_user_emb, v_embs.t()).squeeze(dim=-1).detach()
            scores = (id_scores + t_scores + v_scores) / 3

            for user_id, label, score, id_score, t_score, v_score in zip(
                user_ids, labels, scores, id_scores, t_scores, v_scores
            ):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                (
                    score[history],
                    id_score[history],
                    t_score[history],
                    v_score[history],
                ) = (-np.inf, -np.inf, -np.inf, -np.inf)
                score, id_score, t_score, v_score = (
                    score[1:],
                    id_score[1:],
                    t_score[1:],
                    v_score[1:],
                )

                eval_id.append(
                    metrics_topK(id_score, label, item_rank, topK, local_rank)[1]
                )
                eval_text.append(
                    metrics_topK(t_score, label, item_rank, topK, local_rank)[1]
                )
                eval_image.append(
                    metrics_topK(v_score, label, item_rank, topK, local_rank)[1]
                )
                rank, res = metrics_topK(score, label, item_rank, topK, local_rank)
                rank_list.append(rank.detach().cpu())
                user_list.append(user_id)
                item_list.append(pop_prob_list[eval_seq[user_id][-1]])
                eval_all.append(res)

        eval_all = torch.stack(tensors=eval_all, dim=0).t().contiguous()
        eval_id = torch.stack(tensors=eval_id, dim=0).t().contiguous()
        eval_text = torch.stack(tensors=eval_text, dim=0).t().contiguous()
        eval_image = torch.stack(tensors=eval_image, dim=0).t().contiguous()
        eval_dict = {"te": eval_text, "im": eval_image, "id": eval_id, "fu": eval_all}
        for mode, eval_results in eval_dict.items():
            Hit10, nDCG10 = eval_results
            mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
            print_metrics(mean_eval, Log_file, v_or_t, mode)

        np.save(
            "./results/{}/rank_list_{}.npy".format(args.dataset, epoch - 1),
            np.array(rank_list),
        )
        np.save(
            "./results/{}/user_list_{}.npy".format(args.dataset, epoch - 1),
            np.array(user_list),
        )
        np.save(
            "./results/{}/item_list_{}.npy".format(args.dataset, epoch - 1),
            np.array(item_list),
        )

    return mean_eval[0], mean_eval[1]
