from glob import glob
from multiprocessing import Pool
import librosa
import numpy as np
import soundfile as sf
from itertools import groupby
import os


def get_key(path):
    elements = path.split("/")[-1].replace("-", "_").split("_")
    return elements[1]+"_"+elements[-1]


def mix_wav(input_files, output_file):

    array1, sr1 = sf.read(input_files[0])
    array2, sr2 = sf.read(input_files[1])

    if sr1 == sr2:
        sf.write(output_file, array1+array2[0:len(array1)], sr2)


def mix_wav_with_shift(input_files, output_file):

    array1, sr1 = sf.read(input_files[0])
    array2, sr2 = sf.read(input_files[1])

    if sr1 != 16000:
        array1 = librosa.resample(
            array1, orig_sr=sr1, target_sr=16000, res_type="fft")
    if sr2 != 16000:
        array2 = librosa.resample(
            array2, orig_sr=sr2, target_sr=16000, res_type="fft")

    if len(array1) == len(array2):
        print("{} : {}".format(input_files[0].split("/")[-1], 0), flush=True)
        sf.write(output_file, array1+array2, 16000)

    else:
        (file_large, array_large), (file_small, array_small) = sorted([(input_files[0].split(
            "/")[-1], array1), (input_files[1].split("/")[-1], array2)], key=lambda x: len(x[1]), reverse=True)

        max_abs_sum = 0
        length_diff = len(array_large) - len(array_small)

        if length_diff > 160000:
            return

        # for shift in range(-80000, length_diff+80001, 500):
        for shift in [0]:
            mixed_array = np.concatenate((np.zeros(max(-shift, 0)),
                                          array_large[max(0, shift):min(
                                              len(array_large), len(array_small) + shift)],
                                          np.zeros(max(0, shift-length_diff))),
                                         axis=None) + array_small

            shifted_abs_sum = np.sum(np.absolute(mixed_array))
            shifted_abs_sum = shifted_abs_sum

            if shifted_abs_sum > max_abs_sum:
                max_abs_sum = shifted_abs_sum
                optimal_shift = shift
                optimal_mixed_array = mixed_array

        print("{} : {}".format(file_large, optimal_shift/16000), flush=True)
        sf.write(output_file, optimal_mixed_array, 16000)


def mix_wavs(path_pattern, partition, workers=5):

    audios = glob("./_data_in_processing/imda_raw/IMDA_-_National_Speech_Corpus/PART3/Audio_Separate_StandingMic/*.wav", recursive=True)
    audios.sort(key=get_key)

    pool = Pool(processes=workers)
    for i, (key, value) in enumerate(groupby(audios, key=get_key)):
        input_files = list(value)
        if not len(input_files) == 2:
            print(input_files, flush=True)
        else:
            output_file = "mixed_wav/{}/".format(partition)+key
            if os.path.exists(output_file):
                continue
            pool.apply_async(func=mix_wav, kwds={
                             "input_files": input_files, "output_file": output_file})

    pool.close()
    pool.join()
    print('结束', flush=True)


if __name__ == "__main__":

    print(os.getpid(), flush=True)

    mix_wav(
        [
            "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/IMDA_-_National_Speech_Corpus/PART3/Audio_Separate_IVR/conf_2737_2737_00862530.wav",
            "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/IMDA_-_National_Speech_Corpus/PART3/Audio_Separate_IVR/conf_2737_2737_00862684.wav"
        ],
        "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda_raw/IMDA_-_National_Speech_Corpus/PART3/audio_separate/conf_2737_2737.wav"
    )
    mix_wavs()
    print("complete", flush=True)
