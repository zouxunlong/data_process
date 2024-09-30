
import html
import itertools
from sentsplit.segment import SentSplit


sentence_splitter_en = SentSplit('en', strip_spaces=True, maxcut=256)
sentence_splitter_zh = SentSplit('zh', strip_spaces=True, maxcut=100)


def sentence_split(language_type, text):

    text = html.unescape(text.strip())

    sents = [' '.join(sent.split()) for sent in text.split('\n') if sent.strip()]

    if language_type in {'en', 'ms'}:
        return list(itertools.chain.from_iterable(sentence_splitter_en.segment(sents)))
    if language_type in {'zh'}:
        return list(itertools.chain.from_iterable(sentence_splitter_zh.segment(sents)))


def sent_split(text):
    from datasets import load_from_disk
    ds=load_from_disk("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/zh.hf")
    for sample in ds:
        with open("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/zh.txt", 'a', encoding='utf8') as file_out:
            for line in sentence_split("zh", sample['text']):
                file_out.write(line+'\n')
            file_out.write('\n')
