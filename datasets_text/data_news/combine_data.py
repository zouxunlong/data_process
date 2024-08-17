import os


local_sources=[
    "Bernama", 
    "citynomads", 
    "CNA", 
    "Mothership", 
    "straitstimes", 
    "Theindependent",
    "Weekender",
    "8world",
    "zaobao",
    ]


# with open("en.local.jsonl", "a", encoding="utf8") as f_en_local, \
#     open("en.non_local.jsonl", "a", encoding="utf8") as f_en_nonlocal, \
#     open("id.jsonl", "a", encoding="utf8") as f_id, \
#     open("ms.jsonl", "a", encoding="utf8") as f_ms, \
#     open("ta.jsonl", "a", encoding="utf8") as f_ta, \
#     open("th.jsonl", "a", encoding="utf8") as f_th, \
#     open("vi.jsonl", "a", encoding="utf8") as f_vi, \
#     open("zh.local.jsonl", "a", encoding="utf8") as f_zh_local, \
#     open("zh.non_local.jsonl", "a", encoding="utf8") as f_zh_nonlocal:

with open("zh.local.jsonl", "a", encoding="utf8") as f_zh_local:
    for root_dir, dirs, files in os.walk("news_article"):
        dirs.sort()
        files.sort()
        for file in files:
            lang, source=root_dir.split("/")[-1].split("_")
            # if lang in ["en"]:
            #     if source in local_sources:
            #         f_en_local.write(open(os.path.join(root_dir, file)).read())
            #     else:
            #         f_en_nonlocal.write(open(os.path.join(root_dir, file)).read())
            if lang in ["zh"]:
                if source in local_sources:
                    f_zh_local.write(open(os.path.join(root_dir, file)).read())
                # else:
                #     f_zh_nonlocal.write(open(os.path.join(root_dir, file)).read())
            # if lang in ["id"]:
            #     f_id.write(open(os.path.join(root_dir, file)).read())
            # if lang in ["ms"]:
            #     f_ms.write(open(os.path.join(root_dir, file)).read())
            # if lang in ["ta"]:
            #     f_ta.write(open(os.path.join(root_dir, file)).read())
            # if lang in ["th"]:
            #     f_th.write(open(os.path.join(root_dir, file)).read())
            # if lang in ["vi"]:
            #     f_vi.write(open(os.path.join(root_dir, file)).read())
        print("complete {}".format(root_dir), flush=True)
    print("complete all", flush=True)
    

