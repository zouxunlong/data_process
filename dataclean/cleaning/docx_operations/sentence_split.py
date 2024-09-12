import os
from sentsplit.segment import SentSplit


sent_splitter_en = SentSplit('en', maxcut=80, strip_spaces=True)
sent_splitter_zh = SentSplit('zh', maxcut=28, strip_spaces=True)


def extract_en_sentences(paragraph):
    sentences = sent_splitter_en.segment(paragraph)
    return sentences


def extract_zh_sentences(paragraph):
    sentences = sent_splitter_zh.segment(paragraph)
    return sentences


def main():

    sentences_src = extract_en_sentences(
        'The majority of road users are law abiding（奉公守法）, but there is still a minority of errant road users（违例公路使用者/害群之马）, both cyclists and motorists. With social media （社交媒体），their bad behaviour often goes viral (爆红). ')
    sentences_tgt = extract_zh_sentences(
        'The majority of road users are law abiding（奉公守法）, but there is still a minority of errant road users（违例公路使用者/害群之马）, both cyclists and motorists. With social media （社交媒体），their bad behaviour often goes viral (爆红). ')

    text_list_dict = {'en': sentences_src, 'id': sentences_tgt}


if __name__ == "__main__":
    main()
