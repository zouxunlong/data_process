import os
import time


start_time = time.time()


def rename_file(filepath):
    os.rename(filepath, os.path.splitext(filepath)[0])


rootdir = '/home/xuanlong/dataclean/data/VMT'
for root, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith('.select'):
            rename_file(os.path.join(root, file))

print("--- %s seconds ---" % (time.time() - start_time))
