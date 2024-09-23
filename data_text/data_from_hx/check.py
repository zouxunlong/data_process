import json

with open("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc.id.jsonl") as file_in, \
    open("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc.id.out.jsonl", "w", encoding="utf8") as file_out:
    for line in file_in:
        item=json.loads(line)
        file_out.write(json.dumps({"text":item["text"], "language": "id"}, ensure_ascii=False)+"\n")
print("complete id", flush=True)


with open("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc.zh.jsonl") as file_in, \
    open("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc.zh.out.jsonl", "w", encoding="utf8") as file_out:
    for line in file_in:
        item=json.loads(line)
        file_out.write(json.dumps({"text":item["text"], "language": "zh"}, ensure_ascii=False)+"\n")
print("complete zh", flush=True)


with open("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc.en.jsonl") as file_in, \
    open("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_from_hx/sampled_seamc.en.out.jsonl", "w", encoding="utf8") as file_out:
    for line in file_in:
        item=json.loads(line)
        file_out.write(json.dumps({"text":item["text"], "language": "en"}, ensure_ascii=False)+"\n")
print("complete en", flush=True)