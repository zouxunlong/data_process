import json


original_test_dict={}
current_test_dict={}
original_train_dict={}
current_train_dict={}

original_test_lines  = open('pre_ready_datasets/xunlong_working_repo/test_0.txt').readlines()
current_test_lines   = open('pre_ready_datasets/xunlong_working_repo/test_1.txt').readlines()
original_train_lines = open('pre_ready_datasets/xunlong_working_repo/train_0.txt').readlines()
current_train_lines  = open('pre_ready_datasets/xunlong_working_repo/train_1.txt').readlines()

def build_dict(lines):
    value=set()
    set_dict={}
    for line in lines:
        if line.strip() in ["ASR", "SQA", "ASQA", "ST", "SI", "Paralingual"]:
            key=line.strip()
            value.clear()
        elif line.strip():
            value.add(line.strip())
        else:
            set_dict[key]=value.copy()
    return set_dict

original_test_dict=build_dict(original_test_lines)
current_test_dict=build_dict(current_test_lines)
original_train_dict=build_dict(original_train_lines)
current_train_dict=build_dict(current_train_lines)

difference_test_dict={}
difference_train_dict={}

for key in original_test_dict.keys():
    difference_test_dict[key]=list(original_test_dict[key]-current_test_dict[key])

for key in original_train_dict.keys():
    difference_train_dict[key]=list(original_train_dict[key]-current_train_dict[key])

json.dump(difference_test_dict, open("test_2.json", "w"), indent=4)
json.dump(difference_train_dict, open("train_2.json", "w"), indent=4)


