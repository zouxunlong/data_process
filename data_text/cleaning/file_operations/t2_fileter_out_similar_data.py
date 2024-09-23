import os
import time
from simhash import Simhash, SimhashIndex


def filter_most_similar(lines):

    en_sentences = lines
    objs = [(str(k), Simhash(v)) for k, v in enumerate(en_sentences)]
    index = SimhashIndex(objs, k=15)

    seen_index = []
    selected_lines = []

    for obj in objs:

        if obj[0] in seen_index:
            continue
        near_dups=index.get_near_dups_after_index_not_in_seen(obj[0], obj[1], seen_index)
        seen_index.extend(near_dups)
        selected_lines.append(lines[int(obj[0])])
    return selected_lines


def filter_similar(file_path):

    print("start " + file_path, flush=True)

    with open(file_path) as f_in, \
            open(file_path + '.filtered', 'w', encoding='utf8') as f_out:
        lines = []
        for (i, line) in enumerate(f_in):
            lines.append(line)
            if (i+1) % 1000 == 0:
                selected_lines = filter_most_similar(lines)
                f_out.write(''.join(selected_lines))
                lines.clear()
        selected_lines = filter_most_similar(lines)
        f_out.write(''.join(selected_lines))
        lines.clear()
    print("f_inished " + file_path, flush=True)


def main(file_path):
    filter_similar(file_path)


if __name__ == '__main__':
    start_time = time.time()
    file_path = '/home/xuanlong/dataclean/data/Batch8(CD8)_extracted_combined.en-ms'
    main(file_path)
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
