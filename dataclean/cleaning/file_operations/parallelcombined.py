

with open('/home/xuanlong/dataclean/data/cleaned/clean_sorted1.en-zh', 'w', encoding='utf8') as f_out, \
        open('/home/xuanlong/dataclean/data/cleaned/clean_sorted3.en-zh.en', 'r', encoding='utf8') as f_in_en, \
        open('/home/xuanlong/dataclean/data/cleaned/clean_sorted3.en-zh.zh', 'r', encoding='utf8') as f_in_zh:
    for (i, sentence_en), (j, sentence_zh) in zip(enumerate(f_in_en), enumerate(f_in_zh)):
        f_out.write(sentence_en.strip()+'\t'+sentence_zh.strip()+'\n')
