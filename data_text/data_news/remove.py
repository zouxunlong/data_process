import os

for root_dir, dirs, files in os.walk("data_news/news_article"):
    for file in files:
        if not file.endswith(".jsonl") or file.endswith(".en.jsonl"):
            os.remove(os.path.join(root_dir, file))
            

