from glob import glob
from multiprocessing import Pool
import os
import librosa
import soundfile as sf


def resample(audio_file):

    array, sr = sf.read(audio_file)

    if sr != 16000:
        print("start resample {}".format(audio_file.split("/")[-1]), flush=True)

        array = librosa.resample(array, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
        sf.write(audio_file, array, 16000)

        print("complete resample {}".format(audio_file.split("/")[-1]), flush=True)

    else:
        print("no need to resample {}".format(audio_file.split("/")[-1]), flush=True)
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
    resample("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/PART5/_error/app_4118_6236_phnd_deb-3.wav")
    print("complete", flush=True)
    print("------take {} seconds---".format(time.time()-start), flush=True)

