import os
from sentsplit.segment import SentSplit


sent_splitter = SentSplit('en', strip_spaces=True)


def extract_sentences(paragraph):
    sentences = sent_splitter.segment(paragraph)
    return sentences


def main():
    paragraph="Aku merasa telanjang ./ Setiap orang untuk dirinya sendiri!"
    sentences=extract_sentences(paragraph)
    print(sentences)


if __name__ == "__main__":
    main()
