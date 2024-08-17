from datasets import load_from_disk
from pprint import pprint
from itertools import chain
import json


def check_sample(dataset_path):
    dataset = load_from_disk(dataset_path)

    with open("instructions.openhermes.txt", "w", encoding="utf8") as f_out:
        # f_out.write("".join(list(chain(*[["\n{}-------------------------\n".format(i)+json.dumps(conv) for conv in conv_list] for i, conv_list in enumerate(dataset["conversations"])]))))
        f_out.write("".join(["\n{}: input-------------------------\n".format(i)+sample["input"]+
                             "\n{}: instruction-------------------------\n".format(i)+sample["instruction"]+
                              "\n{}: output-------------------------\n".format(i)+sample["output"] for i, sample in enumerate(dataset)]))


if __name__ == "__main__":
    dataset_path = "instruction/openhermes"
    check_sample(dataset_path)
