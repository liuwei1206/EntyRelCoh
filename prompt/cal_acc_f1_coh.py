import os
import json
import regex
import re

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from preprocessing_discourse.utils import write_to_file_json, read_from_file_json
from preprocessing_coherence.utils import label2idx

def extract_res_with_keyword(response, keyword):
    pattern = r"<{}>(.*?)</{}>".format(keyword, keyword)
    matches = re.findall(pattern, response, re.DOTALL)
    if matches is None or len(matches) == 0:
        # print("++++", response)
        return "low"
    else:
        return matches[-1].strip()

def metric_for_file(file_path):
    data_list = read_from_file_json(file_path, json_type=2)
    pred_ids = []
    label_ids = []
    # label2idx = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    for data in data_list:
        if "<answer>" in data["label"]:
            pred = extract_res_with_keyword(data["predict"], "answer")
            label = extract_res_with_keyword(data["label"], "answer")
        else:
            pred = data["predict"]
            label = data["label"]

        if pred in label2idx:
            pred_ids.append(label2idx[pred])
        else:
            if "high" in pred:
                pred = "high"
            elif "medium" in pred:
                pred = "medium"
            else:
                pred = "low"
            pred_ids.append(label2idx[pred])
        # pred_ids.append(label2idx[pred])
        label_ids.append(label2idx[label])

    assert len(pred_ids) == len(label_ids), (len(pred_ids), len(label_ids))

    acc = accuracy_score(y_true=label_ids, y_pred=pred_ids)
    f1 = f1_score(y_true=label_ids, y_pred=pred_ids, average="macro")

    """
    res = classification_report(
        y_true=label_ids,
        y_pred=pred_ids,
        target_names=["low", "medium", "high"],
        digits=4
    )
    print(res)
    """

    return acc, f1

if __name__ == "__main__":
    ## 1 zero-shot
    llm = "Llama-3.1-8B-Instruct"
    llm = "Llama-3.3-70B-Instruct"
    mode = "entyrel"
    # """
    for task in ["toefl_p1"]: # ["clinton", "enron", "yahoo", "yelp", "toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8", "cohe"]:
        fold_id = 1
        file_path = "saves/{}/lora/eval_{}_0shot_{}_1/generated_predictions.jsonl".format(llm, task, mode)
        acc, f1 = metric_for_file(file_path)
        print("0shot, task=%s, mode=%s: Acc=%.4f, F1=%.4f"%(task, mode, acc, f1))
    # """

    ## 2 lora
    llm = "Llama-3.1-8B-Instruct"
    mode = "textonly"
    """
    for task in ["yelp"]: # ["clinton", "enron", "yahoo", "yelp"]: # ["toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8"]:
        if task in ["clinton", "enron", "yahoo", "yelp"]:
            fold_num = 10
        else:
            fold_num = 5

        for fold_id in range(3, 4):
            file_path = "saves/{}/lora/eval_{}_{}_{}/generated_predictions.jsonl".format(llm, task, fold_id, mode)
            acc, f1 = metric_for_file(file_path)
            print("lora, task=%s, fold=%d, mode=%s: Acc=%.4f, F1=%.4f"%(task, fold_id, mode, acc, f1))

    """

    """
    llm = "Llama-3.1-8B-Instruct"
    mode = "textrel"
    for task in ["toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p6", "toefl_p7", "toefl_p8"]:
        fold_id = 2
        file_path = "saves/{}/lora/eval_toefl5_{}_{}_{}/generated_predictions.jsonl".format(llm, task, fold_id, mode)
        acc, f1 = metric_for_file(file_path)
        print("lora, task=%s, fold=%d, mode=%s: Acc=%.4f, F1=%.4f"%(task, fold_id, mode, acc, f1))
    """

    """
    llm = "Llama-3.1-8B-Instruct"
    mode = "entyrel"
    for task in ["clinton", "yahoo", "yelp"]:
        fold_id = 2
        file_path = "saves/{}/lora/eval_enron_{}_{}_{}/generated_predictions.jsonl".format(llm, task, fold_id, mode)
        acc, f1 = metric_for_file(file_path)
        print("lora, task=%s, fold=%d, mode=%s: Acc=%.4f, F1=%.4f"%(task, fold_id, mode, acc, f1))
    """

