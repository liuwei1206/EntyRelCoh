# date=2024-08-18
# author=liuwei

import math
import os
import json

import numpy as np
import torch
import random
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import gelu

from module import Embedding, Absolute_Position_Embedding, FusionTransformer
from module import Doc_Pooler, Relative_Position_Embedding

class BaseClassifer(nn.Module):
    def __init__(self, args):
        super(BaseClassifer, self).__init__()

        self.num_labels = args.num_labels
        self.fc1 = nn.Linear(args.input_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size // 4)
        self.classifier = nn.Linear(args.hidden_size // 4, self.num_labels)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc1.weight.data.normal_(mean=0.0, std=0.02)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.normal_(mean=0.0, std=0.02)
        self.fc2.bias.data.zero_()
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        doc_vectors,
        labels=None,
        flag="Train"
    ):
        if doc_vectors.dtype == torch.bfloat16:
            doc_vectors = doc_vectors.float()
        input_vectors = self.dropout(doc_vectors)
        input_vectors = input_vectors.float()
        input_vectors = self.fc1(input_vectors)
        input_vectors = self.dropout(input_vectors)
        input_vectors = self.fc2(input_vectors)
        input_vectors = self.dropout(input_vectors)
        logits = self.classifier(input_vectors)

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class SentTransformer(nn.Module):
    def __init__(self, args):
        super(SentTransformer, self).__init__()

        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.num_labels = args.num_labels

        self.proj = nn.Linear(self.input_size, self.hidden_size)
        self.abs_position_embedding = Absolute_Position_Embedding(
            self.hidden_size, learnable=False
        )
        self.transformer = Transformer_Encoder(
            {
                "num_layers": 1, "hidden_size": args.hidden_size,
                "num_heads": 8, "scaled": True
            }
        )
        self.fc = nn.Linear(args.hidden_size, args.hidden_size // 4)
        self.classifier = nn.Linear(args.hidden_size // 4, self.num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        sent_vectors,
        sent_mask,
        labels,
        flag = "Train"
    ):
        batch_size = sent_vectors.size(0)
        hidden_size = sent_vectors.size(2)

        sent_vectors = self.proj(sent_vectors)
        input_vectors = self.abs_position_embedding(sent_vectors)
        outputs = self.transformer(input_vectors, sent_mask)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class FusionClassifier(nn.Module):
    def __init__(self, args):
        super(FusionClassifier, self).__init__()

        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.embed_dim = args.embed_dim
        self.max_node_len = args.max_sent_num + args.max_rel_num + 1
        self.dropout = nn.Dropout(args.dropout)
        self.num_labels = args.num_labels
        self.rel_embedding = Embedding(
            vocab=args.rel_list,
            embed_dim=args.embed_dim
        )
        self.sent_proj = nn.Linear(self.input_size, self.hidden_size)
        self.rel_proj = nn.Linear(self.embed_dim, self.hidden_size)
        self.rel_pos_embedding = Relative_Position_Embedding(
            self.hidden_size, self.max_node_len*2
        )
        self.fusion_transformer = FusionTransformer(
            {
                "num_layers": 1, "hidden_size": args.hidden_size,
                "num_heads": 8, "scaled": True
            }
        )
        self.fc = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.classifier = nn.Linear(args.hidden_size // 4, self.num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        sent_vectors,
        sent_mask,
        rel_ids,
        pos_start,
        pos_end,
        sent_rel_mask,
        labels=None,
        flag="Train"
    ):
        # for sent
        if sent_vectors.dtype == torch.bfloat16:
            sent_vectors = sent_vectors.float()
        sent_vectors = self.dropout(sent_vectors)
        sent_vectors = self.sent_proj(sent_vectors)

        # for rel
        rel_vectors = self.rel_embedding(rel_ids)
        rel_vectors = self.dropout(rel_vectors)
        rel_vectors = self.rel_proj(rel_vectors)

        input_vectors = torch.cat(
            (sent_vectors, rel_vectors), dim=1
        )
        input_vectors = self.dropout(input_vectors)

        # pos embedding
        rel_pos_vectors = self.rel_pos_embedding(
            pos_start, pos_end
        )

        output = self.fusion_transformer(
            hidden_states=input_vectors,
            attention_mask=sent_mask,
            fusion_mask=sent_rel_mask,
            rel_pos_input=rel_pos_vectors
        )

        output = self.dropout(output)
        output = self.fc(output)
        output = self.dropout(output)
        logits = self.classifier(output)

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

