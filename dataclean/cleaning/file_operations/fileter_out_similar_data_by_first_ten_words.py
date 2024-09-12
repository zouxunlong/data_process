import os
import time


def filter_similar_line(line, sentence_pair_set,en_sent_first_ten_words_set):

    sentences = line.strip().split('|')
    if len(sentences) != 3:
        return

    score = sentences[0].strip()
    en_sent = sentences[1].strip()
    non_en_sent = sentences[2].strip()
    en_sent_first_ten_words=' '.join(en_sent.split()[:10])

    if en_sent_first_ten_words in en_sent_first_ten_words_set:
        return

    en_sent_first_ten_words_set.add(en_sent_first_ten_words)
    sentence_pair_set.add((score, en_sent, non_en_sent))


def filter(file_path):
    with open(file_path) as fIN:
        sentence_pair_set = set()
        en_sent_first_ten_words_set = set()
        for (i, line) in enumerate(fIN):
            filter_similar_line(line, sentence_pair_set,en_sent_first_ten_words_set)

            if (i+1) % 100000 == 0:
                with open(file_path + '.filtered2', 'a', encoding='utf8') as fOUT:
                    sentence_pair_list = list(sentence_pair_set)
                    sentence_pair_list_sorted = sorted(
                        sentence_pair_list, reverse=True)
                    for score, en_sent, non_en_sent in sentence_pair_list_sorted:
                        fOUT.write("{} | {} | {}\n".format(
                            score, en_sent, non_en_sent))
                sentence_pair_set.clear()
                en_sent_first_ten_words_set.clear()

        with open(file_path + '.filtered2', 'a', encoding='utf8') as fOUT:
            sentence_pair_list = list(sentence_pair_set)
            sentence_pair_list_sorted = sorted(
                sentence_pair_list, reverse=True)
            for score, en_sent, non_en_sent in sentence_pair_list_sorted:
                fOUT.write("{} | {} | {}\n".format(
                    score, en_sent, non_en_sent))
        sentence_pair_set.clear()
        en_sent_first_ten_words_set.clear()

    print("finished " + file_path, flush=True)


def main(rootdir):
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if os.path.splitext(file)[1] in {'.filtered'}:
                file_path = os.path.join(root, file)
                filter(file_path)


if __name__ == '__main__':
    start_time = time.time()
    rootdir = '/home/xuanlong/dataclean/data/parallel/en-ms'
    main(rootdir)
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
