import os
import time
import random


def select(file_path):
    with open(file_path, encoding='utf8') as f_in, open(file_path + '.sampled', 'w', encoding='utf8') as f_out:

        randomlist = random.sample(range(6000000), 2000)
        print(randomlist)

        for i,line in enumerate(f_in):
            if i in randomlist:
                f_out.write(line[9:])
    print("finished " + file_path, flush=True)


def main(rootdir):
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if os.path.splitext(file)[1] in {'.en-zh'}:
                file_path = os.path.join(root, file)
                select(file_path)


if __name__ == '__main__':
    start_time = time.time()
    rootdir = '/home/xuanlong/dataclean/data/parallel_seleted'
    main(rootdir)
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
