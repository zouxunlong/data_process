import re


def preprocess_sentence(sentence):
    s = sentence.lower().strip()
    s = re.sub(r'[^0-9a-zA-Z?.!,:\'’？。！，：：\u4e00-\u9fff]+', r" ", sentence)
    s = re.sub(r'\s+', r" ", s)
    s = s.strip()
    return s


with open('/home/xuanlong/dataclean/data/cleaned/clean_sorted2.en-zh.en', 'r', encoding='utf8') as f_in_en, \
        open('/home/xuanlong/dataclean/data/cleaned/clean_sorted2.en-zh.zh', 'r', encoding='utf8') as f_in_zh, \
        open('/home/xuanlong/dataclean/data/cleaned/clean_sorted3.en-zh.en', 'w', encoding='utf8') as f_out_en, \
        open('/home/xuanlong/dataclean/data/cleaned/clean_sorted3.en-zh.zh', 'w', encoding='utf8') as f_out_zh:
    for (i, sentence_en), (j, sentence_zh) in zip(enumerate(f_in_en), enumerate(f_in_zh)):
        f_out_zh.write(preprocess_sentence(sentence_zh)+'\n')
        f_out_en.write(preprocess_sentence(sentence_en)+'\n')
