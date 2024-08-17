from glob import glob
from multiprocessing import Pool
import os
import librosa
import soundfile as sf


def get_key(path):
    elements = path.split("/")[-1].replace("-", "_").split("_")
    return elements[1]+"_"+elements[-1]


def resample(audio_file):

    array, sr = sf.read(audio_file)

    if sr != 16000:
        print("start resample {}".format(audio_file.split("/")[-1]), flush=True)

        array = librosa.resample(array, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
        sf.write(audio_file, array, 16000)

        print("resampled {}".format(audio_file.split("/")[-1]), flush=True)

    else:
        return


def re_sampling(workers=1):

    audio_files = glob("mixed_wav/PART3/same_room/*.wav", recursive=True)
    audio_files.sort()

    pool = Pool(processes=workers)
    for audio_file in audio_files:
        pool.apply_async(func=resample, kwds={"audio_file":audio_file})

    pool.close()
    pool.join()

    print('结束', flush=True)


if __name__ == "__main__":
    import time
    print(os.getpid(), flush=True)
    start=time.time()
    re_sampling()
    print("complete", flush=True)
    print("------take {} seconds---".format(time.time()-start), flush=True)

