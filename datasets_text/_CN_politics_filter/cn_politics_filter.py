from datasets import load_from_disk
from fire import Fire


def prepare_city_list():
    tmp_list = open("china_city_full_list.txt", encoding="utf-8").readlines()
    tmp_list = [i.strip() for i in tmp_list if len(i.strip()) > 1]
    output_list = []
    for one_item in tmp_list:
        if len(one_item) > 2 and one_item[-1] in ["省", "市", "区", "州", "县"]:
            output_list.append(one_item[:-1])
    output_list = list(set(output_list) | set(tmp_list))
    print(len(output_list))
    print(output_list[:10])
    return output_list


def keyword_counting_filtering(one_text, city_list):
    term_num = 0
    for one_term in ["中央", "党", "镇", "县", "省", "法院", "两会", "疫情", "党员", "公安", "政法", "网信办", "纪委", "涉嫌"]:
        if one_term in one_text:
            term_num += 1
        if term_num > 1:
            break
    if term_num <= 1:
        for one_term in city_list:
            if one_text.count(one_term) > 1:
                term_num += 2
                break
            else:
                if one_term in one_text:
                    term_num += 1
            if term_num > 1:
                break
    if term_num > 1:
        return False
    else:
        return True


def filter(dataset_path, output_path):

    def filter_fn(batch, full_city_list):
        return [True if keyword_counting_filtering(content, full_city_list) else False for content in batch["text"] ]

    full_city_list = prepare_city_list()
    ds = load_from_disk(dataset_path)
    ds = ds.filter(filter_fn, batched=True, num_proc=4, fn_kwargs={"full_city_list": full_city_list})
    ds.save_to_disk("{}.filtered".format(output_path), num_proc=4)


if __name__ == "__main__":

    Fire(filter)

