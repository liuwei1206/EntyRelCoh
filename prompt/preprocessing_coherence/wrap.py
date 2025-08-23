# author=liuwei
# date=2025-04-01

import os
import json
from jinja2 import Template
from collections import defaultdict
import random

from preprocessing_discourse.utils import read_from_file_json, write_to_file_json, read_prompt_file
from utils import data_to_alpace_format_zero, coherence_levels, data_to_alpaca_format, data_to_alpaca_format_withrel, data_to_alpaca_format_sent, data_to_alpace_format_zero_withrel, data_to_alpace_format_zero_sent


def file_to_alpaca_format_eval(
    task_name,
    fold_id,
    shot_num=0
):
    def sample_examples(task_name, fold_id, shot_num):
        train_data_file = "data/raw/{}/{}/test.json".format(task_name, fold_id)
        train_data_list = read_from_file_json(train_data_file, json_type=2)

        # collect all examples per label, then sample
        label_examples = defaultdict(list)
        for data in train_data_list:
            label_examples[coherence_levels[data["score"]]].append(data)
        low_examples = []
        tmp_examples = label_examples["low"]
        ids = [random.randint(0, len(tmp_examples)-1) for _ in range(shot_num)]
        low_examples = [(tmp_examples[idx], "low") for idx in ids]

        medium_examples = []
        tmp_examples = label_examples["medium"]
        ids = [random.randint(0, len(tmp_examples) - 1) for _ in range(shot_num)]
        medium_examples = [(tmp_examples[idx], "medium") for idx in ids]

        high_examples = []
        tmp_examples = label_examples["high"]
        ids = [random.randint(0, len(tmp_examples) - 1) for _ in range(shot_num)]
        high_examples = [(tmp_examples[idx], "high") for idx in ids]

        all_examples = []
        for idx in range(shot_num):
            all_examples.append(low_examples[idx])
            all_examples.append(medium_examples[idx])
            all_examples.append(high_examples[idx])

        return all_examples

    raw_data_file = "data/raw/{}/{}/test.json".format(task_name, fold_id)
    data_list = read_from_file_json(raw_data_file, json_type=2)

    if shot_num == 0:
        instruction_name = "0shot"
        instruction_template_path = "data/templates/0shot.txt"
    else:
        instruction_name = "{}shot".format(shot_num)
        instruction_template_path = "data/templates/fewshot.txt"

    response_template_path = "data/templates/0shot_res.txt"
    instruction_template = Template(open(instruction_template_path, 'r').read())
    response_template = Template(open(response_template_path, 'r').read())

    if shot_num == 0:
        # alpaca_data_list = data_to_alpace_format_zero(
        #     data_list, instruction_template, response_template
        # )
        alpaca_data_list = data_to_alpace_format_zero_withrel(
            data_list, instruction_template, response_template
        )
        # print("+++")
        # alpaca_data_list = data_to_alpace_format_zero_sent(
        #     data_list, instruction_template, response_template
        # )
    else:
        example_list = sample_examples(task_name, fold_id, shot_num)
        # alpaca_data_list = data_to_alpace_format_few(
        #     data_list, example_list,
        #     instruction_template, response_template
        # )
        alpaca_data_list = data_to_alpace_format_few_withrel(
            data_list, example_list,
            instruction_template, response_template
        )

    out_dir = "data/wrapped/{}/{}/{}".format(task_name, fold_id, instruction_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test.json")
    write_to_file_json(out_file, alpaca_data_list)

def file_to_alpaca_format(
    task_name,
    fold_id,
    mode,
    instruction_template_path,
    response_template_path
):
    raw_data_file = "data/raw/{}/{}/{}.json".format(task_name, fold_id, mode)
    data_list = read_from_file_json(raw_data_file, json_type=2)
    instruction_name = instruction_template_path.split("/")[-1].split(".")[0]
    instruction_template = Template(open(instruction_template_path, 'r').read())
    response_template = Template(open(response_template_path, 'r').read())

    alpaca_data_list = data_to_alpaca_format(
        data_list,
        instruction_template,
        response_template
    )
    out_dir = "data/wrapped/{}/{}/{}".format(task_name, fold_id, instruction_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "{}.json".format(mode))
    write_to_file_json(out_file, alpaca_data_list)

def file_to_alpaca_format_withrel(
    task_name,
    fold_id,
    mode,
    instruction_template_path,
    response_template_path,
    no_rel=False,
    no_enty=False,
    max_num=32
):
    raw_data_file = "data/raw/{}/{}/{}.json".format(task_name, fold_id, mode)
    data_list = read_from_file_json(raw_data_file, json_type=2)
    instruction_name = instruction_template_path.split("/")[-1].split(".")[0]
    instruction_template = Template(open(instruction_template_path, 'r').read())
    response_template = Template(open(response_template_path, 'r').read())

    alpaca_data_list = data_to_alpaca_format_withrel(
        data_list,
        instruction_template,
        response_template,
        no_rel,
        no_enty,
        max_num
    )
    out_dir = "data/wrapped/{}/{}/{}".format(task_name, fold_id, instruction_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "{}.json".format(mode))
    write_to_file_json(out_file, alpaca_data_list)


def file_to_alpaca_format_sent(
    task_name,
    fold_id,
    mode,
    instruction_template_path,
    response_template_path
):
    raw_data_file = "data/raw/{}/{}/{}.json".format(task_name, fold_id, mode)
    data_list = read_from_file_json(raw_data_file, json_type=2)
    instruction_name = instruction_template_path.split("/")[-1].split(".")[0]
    instruction_template = Template(open(instruction_template_path, 'r').read())
    response_template = Template(open(response_template_path, 'r').read())

    alpaca_data_list = data_to_alpaca_format_sent(
        data_list,
        instruction_template,
        response_template
    )
    out_dir = "data/wrapped/{}/{}/{}".format(task_name, fold_id, instruction_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "{}.json".format(mode))
    write_to_file_json(out_file, alpaca_data_list)


def zeroshot_dataset(task, no_enty=False, no_rel=False):
    train_file = "data/raw/{}/1/train.json".format(task)
    dev_file = "data/raw/{}/1/dev.json".format(task)
    test_file = "data/raw/{}/1/test.json".format(task)
    train_data_list = read_from_file_json(train_file, json_type=2)
    dev_data_list = read_from_file_json(dev_file, json_type=2)
    test_data_list = read_from_file_json(test_file, json_type=2)
    if task in ["clinton", "enron", "yahoo", "yelp"]:
        # data_list = test_data_list
        data_list = train_data_list + dev_data_list + test_data_list
    else:
        data_list = train_data_list + dev_data_list + test_data_list

    if no_enty and no_rel:
        instruction_template_path = "data/templates/0plain.txt"
        response_template_path = "data/templates/0plain_res.txt"
    else:
        instruction_template_path = "data/templates/0shot.txt"
        response_template_path = "data/templates/0shot_res.txt"
    instruction_template = Template(open(instruction_template_path, 'r').read())
    response_template = Template(open(response_template_path, 'r').read())

    if "toefl" in task:
        max_num = 32
    else:
        max_num = 32

    if no_rel and no_enty:
        # text
        if task == "cohe":
            alpaca_data_list = data_to_alpace_format_zero_sent(
                data_list,
                instruction_template,
                response_template
            )
        else:
            alpaca_data_list = data_to_alpace_format_zero(
                data_list,
                instruction_template,
                response_template
            )
        model = "textonly"
    elif no_rel:
        # TextEnty
        alpaca_data_list = data_to_alpace_format_zero_withrel(
            data_list,
            instruction_template,
            response_template,
            no_rel=True,
            max_num=max_num
        )
        model = "textenty"
    elif no_enty:
        # TextRel
        alpaca_data_list = data_to_alpace_format_zero_withrel(
            data_list,
            instruction_template,
            response_template,
            no_enty=True,
            max_num=max_num
        )
        model = "textrel"
    else:
        # EntyRel
        alpaca_data_list = data_to_alpace_format_zero_withrel(
            data_list,
            instruction_template,
            response_template,
            max_num=max_num
        )
        model = "entyrel"
    out_dir = "data/wrapped/{}/1/0shot/{}".format(task, model)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "test.json")
    write_to_file_json(out_file, alpaca_data_list)


def lora_dataset(task, no_enty=False, no_rel=False):
    if task in ["clinton", "enron", "yahoo", "yelp"]:
        fold_num = 10
    else:
        fold_num = 5

    if no_enty and no_rel:
        instruction_template_path = "data/templates/plain.txt"
        response_template_path = "data/templates/plain_res.txt"
    else:
        instruction_template_path = "data/templates/entyrel.txt"
        response_template_path = "data/templates/entyrel_res.txt"
    instruction_template = Template(open(instruction_template_path, 'r').read())
    response_template = Template(open(response_template_path, 'r').read())

    for fold_id in range(1, fold_num+1):
        train_file = "data/raw/{}/{}/train.json".format(task, fold_id)
        dev_file = "data/raw/{}/{}/dev.json".format(task, fold_id)
        test_file = "data/raw/{}/{}/test.json".format(task, fold_id)
        train_data_list = read_from_file_json(train_file, json_type=2)
        dev_data_list = read_from_file_json(dev_file, json_type=2)
        test_data_list = read_from_file_json(test_file, json_type=2)
        data_dict = {
            "train": train_data_list,
            "dev": dev_data_list,
            "test": test_data_list
        }
        if "toefl" in task:
            max_num = 32
        else:
            max_num = 32

        for mode in ["train", "dev", "test"]:
            data_list = data_dict[mode]
            if no_rel and no_enty:
                # text
                alpaca_data_list = data_to_alpaca_format(
                    data_list,
                    instruction_template,
                    response_template
                )
                model = "textonly"
            elif no_rel:
                # TextEnty
                alpaca_data_list = data_to_alpaca_format_withrel(
                    data_list,
                    instruction_template,
                    response_template,
                    no_enty=False,
                    no_rel=True,
                    max_num=max_num
                )
                model = "textenty"
            elif no_enty:
                # TextRel
                alpaca_data_list = data_to_alpaca_format_withrel(
                    data_list,
                    instruction_template,
                    response_template,
                    no_enty=True,
                    no_rel=False,
                    max_num=max_num
                )
                model = "textrel"
            else:
                # EntyRel
                alpaca_data_list = data_to_alpaca_format_withrel(
                    data_list,
                    instruction_template,
                    response_template,
                    no_enty=False,
                    no_rel=False,
                    max_num=max_num
                )
                model = "entyrel"

            out_dir = "data/wrapped/{}/{}/lora/{}".format(task, fold_id, model)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, "{}.json".format(mode))
            write_to_file_json(out_file, alpaca_data_list)


if __name__ == "__main__":
    ## 0shot
    # textonly
    for task in []: # ["clinton", "enron", "yahoo", "yelp", "toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8", "cohe"]:
        zeroshot_dataset(task, no_enty=True, no_rel=True)

    # textenty
    for task in ["toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8"]: # ["clinton", "enron", "yahoo", "yelp", "toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8", "cohe"]:
        zeroshot_dataset(task, no_enty=False, no_rel=True)

    # textrel
    for task in ["toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8"]: # ["clinton", "enron", "yahoo", "yelp", "toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8", "cohe"]:
        zeroshot_dataset(task, no_enty=True, no_rel=False)

    # entyrel
    for task in ["toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8"]: # ["clinton", "enron", "yahoo", "yelp", "toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8", "cohe"]:
        zeroshot_dataset(task, no_enty=False, no_rel=False)


    ## for lora
    # textonly
    for task in []: # ["clinton", "enron", "yahoo", "yelp", "toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8", "cohe"]:
        print(task)
        lora_dataset(task, no_enty=True, no_rel=True)

    # textenty
    for task in []: # ["clinton", "enron", "yahoo", "yelp", "toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8", "cohe"]:
        lora_dataset(task, no_enty=False, no_rel=True)

    # textrel
    for task in []: # ["clinton", "enron", "yahoo", "yelp", "toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8", "cohe"]:
        lora_dataset(task, no_enty=True, no_rel=False)

    # entyrel
    for task in []: # ["clinton", "enron", "yahoo", "yelp", "toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8", "cohe"]:
        lora_dataset(task, no_enty=False, no_rel=False)













































































    ## 1. for evaluation, no tuning
    # gcdc
    """
    fold_id = 1
    for task_name in ["enron", "yahoo", "yelp"]:
        for shot_num in [0, 3]:
            file_to_alpaca_format_eval(
                task_name, fold_id, shot_num
            )
    """

    # toefl
    """
    for task_name in ["toefl_p1"]:
        for fold_id in range(1, 2):
            for shot_num in [2]:
                file_to_alpaca_format_eval(
                    task_name, fold_id, shot_num
                )
    """

    ## 2. for tuning
    template_name = "plain"
    instruction_template_path = "data/templates/{}.txt".format(template_name)
    response_template_path = "data/templates/{}_res.txt".format(template_name)

    # gcdc
    """
    for task_name in ["clinton", "enron", "yahoo", "yelp"]:
        for fold_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for mode in ["train", "dev", "test"]:
                file_to_alpaca_format(
                    task_name, fold_id, mode,
                    instruction_template_path,
                    response_template_path
                )
    """

    # toefl
    """
    for task_name in ["toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8"]:
        for fold_id in [1, 2, 3, 4, 5]:
            for mode in ["train", "dev", "test"]:
                file_to_alpaca_format(
                    task_name, fold_id, mode,
                    instruction_template_path,
                    response_template_path
                )
    """

    """
    task_name = "cohe"
    fold_id = 1
    for mode in ["train", "dev", "test"]:
        file_to_alpaca_format(
            task_name, fold_id, mode,
            instruction_template_path,
            response_template_path
        )
    """

    ## 3. for graph
    template_name = "entyrel"
    instruction_template_path = "data/templates/{}.txt".format(template_name)
    response_template_path = "data/templates/{}_res.txt".format(template_name)
    no_rel = False
    no_enty = False
    max_num = 32

    # gcdc

    # toefl
    """
    for task_name in ["toefl_p1"]:
        for fold_id in [1]:
            for mode in ["train", "dev", "test"]:
                file_to_alpaca_format_withrel(
                    task_name, fold_id, mode,
                    instruction_template_path,
                    response_template_path,
                    no_rel, no_enty, max_num
                )
    """

    """
    template_name = "sent"
    instruction_template_path = "data/templates/{}.txt".format(template_name)
    response_template_path = "data/templates/{}_res.txt".format(template_name)
    for task_name in ["toefl_p1"]:
        for fold_id in [1]:
            for mode in ["train", "dev", "test"]:
                file_to_alpaca_format_sent(
                    task_name, fold_id, mode,
                    instruction_template_path,
                    response_template_path
                )
    """