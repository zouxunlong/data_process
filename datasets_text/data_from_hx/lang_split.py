import json

en_count=0
cn_count=0
id_count=0
other_count=0

with open("sampled_seamc.jsonl") as f_in, \
    open("sampled_seamc.zh.jsonl", "w", encoding="utf8") as f_zh, \
    open("sampled_seamc.en.jsonl", "w", encoding="utf8") as f_en, \
    open("sampled_seamc.other.jsonl", "w", encoding="utf8") as f_other, \
    open("sampled_seamc.id.jsonl", "w", encoding="utf8") as f_id:
    for i, line in enumerate(f_in):
        if i % 100000 == 0:
            print(i, flush=True)
        item = json.loads(line)
        if item["language"] == "en":
            f_en.write(line)
        elif item["language"] == "cn":
            f_zh.write(line)
        elif item["language"] == "id":
            f_id.write(line)
        else:
            f_other.write(line)
    print("complete", flush=True)

