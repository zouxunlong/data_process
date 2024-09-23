import json

with open("/home/user/data/data_text/data_news/ta.jsonl") as file_in, \
    open("/home/user/data/data_text/data_news/ta.out.jsonl", "w", encoding="utf8") as file_out:
    for line in file_in:
        item=json.loads(line)
        file_out.write(json.dumps({"date":item["date"], "source": item["source"], 
                                   "title": item["title"], "text": item["text"]}, ensure_ascii=False)+"\n")
print("complete")
