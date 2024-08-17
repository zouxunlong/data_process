from glob import glob
from itertools import groupby
from multiprocessing import Pool
import os
import librosa
import soundfile as sf
import numpy as np


def get_key(path):
    elements = path.split("/")[-1].replace("-", "_").split("_")
    return elements[1]+"_"+elements[-1]


def select_better(input_files, output_file):

    array1, sr1 = sf.read(input_files[0])
    array2, sr2 = sf.read(input_files[1])

    if sr1 != 16000:
        print("error at {}".format(input_files[0]), flush=True)
        # array1 = librosa.resample(array1, orig_sr=sr1, target_sr=16000, res_type="kaiser_best")
    if sr2 != 16000:
        print("error at {}".format(input_files[1]), flush=True)
        # array2 = librosa.resample(array2, orig_sr=sr2, target_sr=16000, res_type="kaiser_best")

    std1 = np.std(np.absolute(array1))
    std2 = np.std(np.absolute(array2))

    if std1 < std2:
        # sf.write(output_file, array1, 16000)
        print("{} : {}".format(input_files[0].split("/")[-1], 0), flush=True)
    else:
        # sf.write(output_file, array2, 16000)
        print("{} : {}".format(input_files[1].split("/")[-1], 0), flush=True)


def select_betters(workers=16):

    audios = glob("Same_Room_Audio/*.wav", recursive=True)
    audios.sort(key=get_key)

    pool = Pool(processes=workers)
    for key, value in groupby(audios, key=get_key):
        input_files = list(value)
        if not len(input_files) == 2:
            print(input_files, flush=True)
        else:
            output_file = "mixed_wav/PART4/same_room/{}".format(key)
            # if os.path.exists(output_file):
            #     continue
            pool.apply_async(func=select_better, kwds={"input_files":input_files,"output_file":output_file})

    pool.close()
    pool.join()
    print('结束', flush=True)


if __name__ == "__main__":
    import time
    print(os.getpid(), flush=True)
    start=time.time()
    select_betters()
    print("complete", flush=True)
    print("------take {} seconds---".format(time.time()-start), flush=True)

