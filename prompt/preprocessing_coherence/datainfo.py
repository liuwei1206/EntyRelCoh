import os
import json

from preprocessing_discourse.utils import read_from_file_json, write_to_file_json

if __name__ == "__main__":
    data_list = {}
    data_list["identify"] = {"file_name": "identity.json"}

    for task in ["clinton", "enron", "yahoo", "yelp", "toefl_p1", "toefl_p2", "toefl_p3", "toefl_p4", "toefl_p5", "toefl_p6", "toefl_p7", "toefl_p8", "cohe"]:
        if task in ["clinton", "enron", "yahoo", "yelp"]:
            fold_num = 10
        else:
            fold_num = 5

        # 0shot
        for mode in ["textonly", "textenty", "textrel", "entyrel"]:
            dataset_name = "{}_0shot_{}".format(task, mode)
            data_list[dataset_name] = {
                "file_name": "{}/1/0shot/{}/test.json".format(task, mode),
                "columns": {
                    "prompt": "instruction",
                    "response": "output",
                    "system": "system_prompt"
                }
            }

        # lora
        for fold_id in range(1, fold_num+1):
            for mode in ["textonly", "textenty", "textrel", "entyrel"]:
                for split in ["train", "dev", "test"]:
                    dataset_name = "{}_{}_{}_{}".format(task, fold_id, mode, split)
                    data_list[dataset_name] = {
                        "file_name": "{}/{}/lora/{}/{}.json".format(task, fold_id, mode, split),
                        "columns": {
                            "prompt": "instruction",
                            "response": "output",
                            "system": "system_prompt"
                        }
                    }

    file_path = "data/cohe_dataset_info.json"
    write_to_file_json(file_path, data_list)