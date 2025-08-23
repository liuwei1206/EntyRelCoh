# date=2024-08-16
# author=liuwei

import os
import json

import numpy as np
import torch
import pickle


def read_text_label(file_name):
    """Read text and label from a json file"""
    texts = []
    labels = []
    raws = []

    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                text = sample["text"]
                label = sample["score"]

                texts.append(text)
                labels.append(label)
                raws.append((text, label))

    return texts, labels, raws


def read_samples(file_name):
    samples = []
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                samples.append(sample)

    return samples


def mask_position_for_fusion(
    max_sent_num, real_sent_num,
    max_rel_num, rel_sent_ids
):
    """
    Args:
        max_sent_num: the maximum number of sents
        real_sent_num: the real num of sents
        rel_sent_ids: [(rel_id, start_sent_id, end_sent_id), ...]

    Return:
        sent_mask:
        rel_sent_mask:
        sent_rel_positions:
    """
    # 1, for sent_mask, used for pooling
    expand_sent_mask = np.zeros(max_sent_num+max_rel_num, dtype=np.int32)
    expand_sent_mask[:real_sent_num] = 1
    sent_mask = np.zeros(max_sent_num, dtype=np.int32)
    sent_mask[:real_sent_num] = 1

    # 2, sent_rel, mask
    real_rel_num = len(rel_sent_ids)
    if len(rel_sent_ids) > max_rel_num:
        rel_sent_ids = rel_sent_ids[:max_rel_num]
        real_rel_num = max_rel_num
    rel_ids = np.zeros(max_rel_num, dtype=np.int32)
    sent_sent = np.expand_dims(sent_mask, axis=0).repeat(max_sent_num, axis=0)
    rel_mask = np.zeros(max_rel_num, dtype=np.int32)
    rel_mask[:real_rel_num] = 1
    rel_rel = np.expand_dims(rel_mask, axis=0).repeat(max_rel_num, axis=0)
    sent_rel = np.zeros((max_sent_num, max_rel_num), dtype=np.int32)
    rel_sent = np.zeros((max_rel_num, max_sent_num), dtype=np.int32)

    for idx, item in enumerate(rel_sent_ids):
        rel_ids[idx] = item[0]
        start_sent_id = item[1]
        end_sent_id = item[2]
        if end_sent_id >= max_sent_num:
            continue

        sent_rel[start_sent_id][idx] = 1
        sent_rel[end_sent_id][idx] = 1
        rel_sent[idx][start_sent_id] = 1
        rel_sent[idx][end_sent_id] = 1
    fusion_sent_mask = np.concatenate((sent_sent, sent_rel), axis=1) # [sent_len, sent_len+rel_len]
    fusion_rel_mask = np.concatenate((rel_sent, rel_rel), axis=1) # [rel_len, sent_len+rel_len]
    sent_rel_mask = np.concatenate((fusion_sent_mask, fusion_rel_mask), axis=0) # [sent_len+rel_len, sent_len+rel_len]
    # sent_rel_mask = np.concatenate((sent_mask, rel_mask))

    # 3. position
    sent_pos = np.arange(0, max_sent_num)
    rel_start_pos = np.arange(0, max_rel_num)
    rel_end_pos = np.arange(0, max_rel_num)
    for idx, item in enumerate(rel_sent_ids):
        start_sent_id = item[1]
        end_sent_id = item[2]

        rel_start_pos[idx] = start_sent_id
        rel_end_pos[idx] = end_sent_id

    merged_start_pos = np.concatenate((sent_pos, rel_start_pos), axis=0)
    merged_end_pos = np.concatenate((sent_pos, rel_end_pos), axis=0)

    return expand_sent_mask, sent_rel_mask, rel_ids, (merged_start_pos, merged_end_pos)


def labels_from_file(label_file):
    label_list = []
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                label_list.append(line.strip().lower())

    return label_list


def build_embedding_of_corpus(embed_file, vocab, embed_dim, saved_file):
    if os.path.exists(saved_file):
        with open(saved_file, "rb") as f:
            corpus_embed = pickle.load(f)
    else:
        word2vec = {}
        matched_num = 0
        with open(embed_file, "r", encoding="utf-8") as f:
            idx = 0
            for line in f:
                line = line.strip()
                if line:
                    item = line.split()
                    if len(item) != (embed_dim+1):
                        continue
                    word = item[0]
                    vector = item[1:]
                    if word in vocab:
                        word2vec[word] = np.array(vector, dtype=np.float)
                        matched_num += 1
                        if matched_num / len(vocab) >= 0.99:
                            break
                    idx += 1
                    if idx % 21800 == 0:
                        print("loading per%d"%(idx / 21800))

        corpus_embed = np.empty([len(vocab), embed_dim])
        scale = np.sqrt(3.0 / embed_dim)
        num_matched = 0
        num_non_matched = 0
        missing_words = []
        for idx, word in enumerate(vocab):
            if word in word2vec:
                corpus_embed[idx, :] = word2vec[word]
                num_matched += 1
            elif word.lower() in word2vec:
                corpus_embed[idx, :] = word2vec[word.lower()]
                num_matched += 1
            else:
                corpus_embed[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
                num_non_matched += 1
                missing_words.append(word)
        print("total: %d, matched: %d, non-matched: %d"%(len(vocab), num_matched, num_non_matched))
        with open(saved_file, "wb") as f:
            pickle.dump(corpus_embed, f)

    return corpus_embed
