import os
from sentsplit.segment import SentSplit

sentence_splitter_en = SentSplit('en', strip_spaces=True, maxcut=512)
sentence_splitter_zh = SentSplit('zh', strip_spaces=True, maxcut=150)


def extract_sentences(file, tgt):

    sent_src=[]
    sent_tgt=[]

    if tgt in["ms", "ta"]:
        sentence_splitter=sentence_splitter_en
    if tgt in["zh"]:
        sentence_splitter=sentence_splitter_zh

    sentences_src, sentences_tgt = [content.split("\n") for content in open(file).read().split("\n\n")]

    for sentence in sentences_src:
        sentences = sentence_splitter_en.segment(sentence.strip())
        sent_src.extend(sentences)

    for sentence in sentences_tgt:
        sentences = sentence_splitter.segment(sentence.strip())
        sent_tgt.extend(sentences)

    return sent_src, sent_tgt


def main():

    files = os.listdir("matched")
    files.sort()
    for file in files:
        tgt = file.split('.')[-1].split("_")[-1]
        sent_src, sent_tgt=extract_sentences("matched/{}".format(file), tgt)
        print("finished {}".format(file), flush=True)



if __name__ == "__main__":
    main()
    print('finished all')
