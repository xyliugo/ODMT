import os
import torch
import numpy as np
from utils.dataset import IdEvalDataset, LmdbEvalDataset
from torch.utils.data import DataLoader


def extract_text_feature(bert_model, item_text, local_rank, args):
    bert_model.to(local_rank)
    item_dataset = IdEvalDataset(data=np.arange(len(item_text)))
    item_dataloader = DataLoader(
        item_dataset, batch_size=256, num_workers=args.num_workers, pin_memory=True
    )
    t_feat = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            inputs = torch.LongTensor(item_text[input_ids]).to(local_rank)
            _, num_words = inputs.shape
            num_words = num_words // 2
            text_ids = torch.narrow(inputs, 1, 0, num_words)
            text_attmask = torch.narrow(inputs, 1, num_words, num_words)
            outputs = bert_model(
                input_ids=text_ids,
                attention_mask=text_attmask,
                output_hidden_states=True,
            )
            outputs = outputs.hidden_states[12]
            t_feat.extend(outputs)

    t_feat = torch.stack(tensors=t_feat, dim=0).to(local_rank)
    print("t_feat shape: {}".format(t_feat.shape))
    return t_feat.detach()


def extract_image_feature(vit_model, item_num, item_id_to_keys, local_rank, args):
    vit_model.to(local_rank)
    item_dataset = LmdbEvalDataset(
        data=np.arange(item_num + 1),
        item_id_to_keys=item_id_to_keys,
        db_path=os.path.join(args.root_data_dir, args.dataset, args.image_data),
        resize=args.image_resize,
        mode="image",
        args=args,
    )
    item_dataloader = DataLoader(
        item_dataset, batch_size=256, num_workers=args.num_workers, pin_memory=True
    )
    v_feat = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            outputs = vit_model(input_ids, output_hidden_states=True)
            outputs = outputs.hidden_states[12]
            v_feat.extend(outputs)

    v_feat = torch.stack(tensors=v_feat, dim=0).to(local_rank)
    print("v_feat shape: {}".format(v_feat.shape))
    return v_feat.detach()
