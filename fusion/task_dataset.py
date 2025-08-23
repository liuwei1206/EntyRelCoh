# author = liuwei
# date = 2025-04-03

import os
import json
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm, trange

from utils import build_graph, mask_position_for_fusion

random.seed(106524)

def get_sent_boundary(sents, tokenizer):
    """get the sent boundary, so we can extract the sent vector"""
    # 1. document
    whole_document = " ".join(sents)

    # 2. for sent boundary
    cur_text = ""
    sent_start_ids = []
    sent_end_ids = []
    for sent in sents:
        if cur_text == "":
            start_pos = 1
        else:
            start_pos = 1 + len(tokenizer.tokenize(cur_text))
        cur_text = cur_text + " " + sent
        # add 1 because the [CLS] token
        end_pos = 1 + len(tokenizer.tokenize(cur_text))
        sent_start_ids.append(start_pos)
        sent_end_ids.append(end_pos)

    res = (whole_document, sent_start_ids, sent_end_ids)

    return res


def inter_sentential_rels(rels, spans, no_rel=False, no_enty=False):
    rel_sent_ids = []
    for rel, span in zip(rels, spans):
        start_sent_id = span[0]
        end_sent_id = span[1]
        if start_sent_id != end_sent_id:
            if no_rel:
                if rel == "entity":
                    rel_sent_ids.append((rel, start_sent_id, end_sent_id))
            elif no_enty:
                if rel != "entity":
                    rel_sent_ids.append((rel, start_sent_id, end_sent_id))
            else:
                rel_sent_ids.append((rel, start_sent_id, end_sent_id))

    return rel_sent_ids


def flatten_emb(emb, emb_mask):
    batch_size = emb.size(0)
    seq_length = emb.size(1)
    flat_emb = emb.view(batch_size*seq_length, -1)
    flat_emb_mask = emb_mask.view(batch_size * seq_length)

    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]


def get_span_representation(span_starts, span_ends, hidden_states, attention_mask):
    """
    refer to: https://github.com/huminghao16/SpanABSA/blob/master/bert/sentiment_modeling.py
    N: batch size
    M: span number
    L: seq length
    D: hidden size
    Args:
        span_starts: [N, M]
        span_ends: [N, M]
        hidden_states: [N, L, D]
        attention_mask: [N, L]
    """
    N = hidden_states.size(0)
    M = span_starts.size(1)
    input_len = torch.sum(attention_mask, dim=-1).to(hidden_states.device)  # [N]
    span_offset = torch.cumsum(input_len, dim=0).to(hidden_states.device)  # [N]
    span_offset -= input_len

    span_starts_offset = span_starts + span_offset.unsqueeze(1)
    span_ends_offset = span_ends + span_offset.unsqueeze(1)
    span_starts_offset = span_starts_offset.view(-1)
    span_ends_offset = span_ends_offset.view(-1)
    span_width = span_ends_offset - span_starts_offset
    max_span_width = torch.max(span_width)

    flat_hidden_states = flatten_emb(hidden_states, attention_mask)  # [<N*L, D], because exclude zero position
    flat_length = flat_hidden_states.size(0)

    # [N*M, max_span_width]
    span_indices = torch.arange(max_span_width).unsqueeze(0).to(hidden_states.device) + span_starts_offset.unsqueeze(1)
    span_indices = torch.min(span_indices, (flat_length - 1) * torch.ones_like(span_indices))  # in case out of boundary
    span_vectors = flat_hidden_states[span_indices, :]  # [N*M, max_span_width, D]
    span_mask = torch.arange(max_span_width).to(hidden_states.device)
    span_mask = span_mask < span_width.unsqueeze(-1)  # [N*M, max_span_width]

    # average the word states to get span representation
    expanded_span_mask = span_mask.unsqueeze(2)  # [N*M, max_span_width, 1]
    masked_span_vectors = span_vectors * expanded_span_mask  # [N*M, max_span_width, D]
    avg_span_vectors = torch.sum(masked_span_vectors, dim=1) / span_width.unsqueeze(1)  # [N*M, D]
    avg_span_vectors = avg_span_vectors.view(N, M, -1)

    return avg_span_vectors


class SentDataset(Dataset):
    def __init__(self, file_name, params):
        self.max_text_length = params["max_text_length"]
        self.max_sent_num = params["max_sent_num"]
        self.rel_list = params["rel_list"]
        self.max_rel_num = params["max_rel_num"]
        self.label_list = params["label_list"]
        self.tokenizer = params["tokenizer"]
        self.encoder = params["encoder"]
        self.pooler_type = params["pooler_type"]
        self.no_rel = params["no_rel"]
        self.no_enty = params["no_enty"]

        assert self.pooler_type in ["avg"], (self.pooler_type)
        assert self.no_enty == False or self.no_rel == False, ("For Enty and Rel, at least one is false!!!")

        mode = file_name.split("/")[-1].split(".")[0]
        dir_name = os.path.dirname(file_name)
        new_dir_name = os.path.join(dir_name, "fusion_vectors")
        os.makedirs(new_dir_name, exist_ok=True)
        if self.no_rel or self.no_enty:
            if self.no_rel:
                if "roberta" in self.encoder.__class__.__name__.lower():
                    saved_file_name = "{}_{}_roberta_{}-{}_norel.pkl".format(
                        self.pooler_type, mode, self.max_text_length, self.max_sent_num
                    )
                elif "llama" in self.encoder.__class__.__name__.lower():
                    saved_file_name = "{}_{}_llama33_{}-{}_norel.pkl".format(
                        self.pooler_type, mode, self.max_text_length, self.max_sent_num
                    )
            elif self.no_enty:
                if "roberta" in self.encoder.__class__.__name__.lower():
                    saved_file_name = "{}_{}_roberta_{}-{}_noenty.pkl".format(
                        self.pooler_type, mode, self.max_text_length, self.max_sent_num
                    )
                elif "llama" in self.encoder.__class__.__name__.lower():
                    saved_file_name = "{}_{}_llama33_{}-{}_noenty.pkl".format(
                        self.pooler_type, mode, self.max_text_length, self.max_sent_num
                    )
        else:
            if "roberta" in self.encoder.__class__.__name__.lower():
                saved_file_name = "{}_{}_roberta_{}-{}.pkl".format(
                    self.pooler_type, mode, self.max_text_length, self.max_sent_num
                )
            elif "llama" in self.encoder.__class__.__name__.lower():
                saved_file_name = "{}_{}_llama33_{}-{}.pkl".format(
                    self.pooler_type, mode, self.max_text_length, self.max_sent_num
                )
        self.np_file = os.path.join(new_dir_name, saved_file_name)

        self.init_dataset(file_name)

    def init_dataset(self, file_name):
        print(self.np_file, "+++++++++")
        if os.path.exists(self.np_file):
            with open(self.np_file, "rb") as f:
                results = pickle.load(f)
                all_doc_vectors = results[0]
                all_sent_vectors = results[1]
                all_sent_mask = results[2]
                all_sent_rel_mask = results[3]
                all_rel_ids = results[4]
                all_start_positions = results[5]
                all_end_positions = results[6]
                all_label_ids = results[7]
        else:
            with open(file_name, "r", encoding="utf-8") as f:
                lines = f.readlines()

                # 1. prepare ids
                all_input_ids = []
                all_attention_mask = []
                all_token_type_ids = []
                all_sent_start_ids = []
                all_sent_end_ids = []
                all_sent_mask = []
                all_sent_rel_mask = []
                all_rel_ids = []
                all_start_positions = []
                all_end_positions = []
                all_label_ids = []

                for line in lines:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        sents = sample["sents"]
                        spans = sample["spans"]
                        rels = sample["rels"]
                        label = sample["score"]

                        if len(sents) == 0:
                            continue
                        sent_res = get_sent_boundary(sents, self.tokenizer)
                        text = sent_res[0]
                        sent_start_ids, sent_end_ids = sent_res[1], sent_res[2]

                        # 2.1 whole document
                        doc_res = self.tokenizer(
                            text=text,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_text_length,
                            return_tensors="pt"
                        )
                        input_ids = doc_res.input_ids
                        attention_mask = doc_res.attention_mask
                        if "token_type_ids" in doc_res:
                            token_type_ids = doc_res["token_type_ids"]
                        else:
                            token_type_ids = torch.zeros_like(attention_mask)

                        # 2.2 sent
                        np_sent_start_ids = np.zeros(self.max_sent_num, dtype=np.int32)
                        np_sent_end_ids = np.ones(self.max_sent_num, dtype=np.int32)
                        sent_mask = np.zeros(self.max_sent_num, dtype=np.int32)
                        real_sent_num = 0
                        if len(sents) > self.max_sent_num:
                            sent_mask[:self.max_sent_num] = 1
                            np_sent_start_ids[:self.max_sent_num] = sent_start_ids[:self.max_sent_num]
                            np_sent_end_ids[:self.max_sent_num] = sent_end_ids[:self.max_sent_num]
                            real_sent_num = self.max_sent_num
                        else:
                            sent_mask[:len(sents)] = 1
                            np_sent_start_ids[:len(sents)] = sent_start_ids
                            np_sent_end_ids[:len(sents)] = sent_end_ids
                            real_sent_num = len(sents)

                        # 3. rel, mask, position
                        rel_sent_ids = inter_sentential_rels(rels, spans, self.no_rel, self.no_enty)
                        rel_sent_ids = [(self.rel_list.index(item[0]), item[1], item[2]) for item in rel_sent_ids]
                        expand_sent_mask, sent_rel_mask, rel_ids, positions = mask_position_for_fusion(
                            self.max_sent_num, real_sent_num, self.max_rel_num, rel_sent_ids
                        )
                        start_positions, end_positions = positions
                        label_id = self.label_list.index(label)

                        all_input_ids.append(input_ids)
                        all_attention_mask.append(attention_mask)
                        all_token_type_ids.append(token_type_ids)
                        all_sent_start_ids.append(torch.tensor(np_sent_start_ids).unsqueeze(0))
                        all_sent_end_ids.append(torch.tensor(np_sent_end_ids).unsqueeze(0))
                        all_sent_mask.append(expand_sent_mask)
                        all_sent_rel_mask.append(sent_rel_mask)
                        all_rel_ids.append(rel_ids)
                        all_start_positions.append(start_positions)
                        all_end_positions.append(end_positions)
                        all_label_ids.append(label_id)

                # 2. generate vectors
                all_doc_vectors = []
                all_sent_vectors = []
                batch_size = 8
                batch_steps = len(all_input_ids) // batch_size + 1
                batch_iter = trange(0, batch_steps, desc="Step")
                for cur_idx in batch_iter:
                    start_pos = cur_idx * batch_size
                    end_pos = (cur_idx + 1) * batch_size
                    if end_pos > len(all_input_ids):
                        end_pos = len(all_input_ids)
                    if start_pos == end_pos:
                        break
                    cur_batch_size = end_pos - start_pos
                    batch_input_ids = torch.cat(
                        all_input_ids[start_pos:end_pos], dim=0
                    ).to(self.encoder.device)
                    batch_attention_mask = torch.cat(
                        all_attention_mask[start_pos:end_pos], dim=0
                    ).to(self.encoder.device)
                    batch_segment_ids = torch.cat(
                        all_token_type_ids[start_pos:end_pos], dim=0
                    ).to(self.encoder.device)
                    batch_sent_start_ids = torch.cat(
                        all_sent_start_ids[start_pos:end_pos], dim=0
                    ).to(self.encoder.device)
                    batch_sent_end_ids = torch.cat(
                        all_sent_end_ids[start_pos:end_pos], dim=0
                    ).to(self.encoder.device)

                    doc_inputs = {
                        "input_ids": batch_input_ids,
                        "attention_mask": batch_attention_mask,
                        # "token_type_ids": batch_segment_ids
                    }
                    with torch.no_grad():
                        doc_outputs = self.encoder(**doc_inputs)
                        seq_doc_outputs = doc_outputs.last_hidden_state
                        batch_doc_length = torch.sum(batch_attention_mask, dim=-1)
                        batch_sum_doc_reps = seq_doc_outputs * batch_attention_mask.unsqueeze(2)
                        batch_sum_doc_reps = torch.sum(batch_sum_doc_reps, dim=1)
                        doc_vectors = batch_sum_doc_reps / batch_doc_length.unsqueeze(1)

                        ## sent vectors
                        # print("## sent ##")
                        sent_vectors = get_span_representation(
                            span_starts=batch_sent_start_ids,
                            span_ends=batch_sent_end_ids,
                            hidden_states=seq_doc_outputs,
                            attention_mask=batch_attention_mask
                        )

                        doc_vectors = doc_vectors.detach().cpu()
                        sent_vectors = sent_vectors.detach().cpu()
                        all_doc_vectors.append(doc_vectors)
                        all_sent_vectors.append(sent_vectors)

                all_doc_vectors = torch.cat(all_doc_vectors, dim=0)
                all_sent_vectors = torch.cat(all_sent_vectors, dim=0)
                all_sent_mask = torch.tensor(np.array(all_sent_mask))
                all_sent_rel_mask = torch.tensor(np.array(all_sent_rel_mask))
                all_rel_ids = torch.tensor(np.array(all_rel_ids))
                all_start_positions = torch.tensor(np.array(all_start_positions))
                all_end_positions = torch.tensor(np.array(all_end_positions))
                all_label_ids = torch.tensor(np.array(all_label_ids))

                # """
                with open(self.np_file, "wb") as f:
                    pickle.dump([
                        all_doc_vectors, all_sent_vectors, all_sent_mask,
                        all_sent_rel_mask, all_rel_ids, all_start_positions,
                        all_end_positions, all_label_ids
                    ], f)
                # """

        self.doc_vectors = all_doc_vectors
        self.sent_vectors = all_sent_vectors
        self.sent_mask = all_sent_mask
        self.sent_rel_mask = all_sent_rel_mask
        self.rel_ids = all_rel_ids
        self.start_positions = all_start_positions
        self.end_positions = all_end_positions
        self.label_ids = all_label_ids
        self.total_size = all_doc_vectors.size(0)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            self.doc_vectors[index],
            self.sent_vectors[index],
            self.sent_mask[index],
            self.sent_rel_mask[index],
            self.rel_ids[index],
            self.start_positions[index],
            self.end_positions[index],
            self.label_ids[index]
        )
