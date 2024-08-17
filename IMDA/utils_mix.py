from glob import glob
from multiprocessing import Pool
import librosa
import numpy as np
import soundfile as sf
from itertools import groupby
import os
import fire

def get_key(path):
    elements = path.split("/")[-1].replace("-", "_").split("_")
    return elements[1]+"_"+elements[-1]


def mix_wav(input_files, output_file):

    array1, sr1 = sf.read(input_files[0])
    array2, sr2 = sf.read(input_files[1])

    if sr1 != 16000:
        array1 = librosa.resample(array1, orig_sr=sr1, target_sr=16000, res_type="fft")
    if sr2 != 16000:
        array2 = librosa.resample(array2, orig_sr=sr2, target_sr=16000, res_type="fft")

    if len(array1)==len(array2):
        print("{} : {}".format(input_files[0].split("/")[-1], 0), flush=True)
        sf.write(output_file, array1+array2, 16000)

    else:
        (file_large, array_large), (file_small, array_small) = sorted([(input_files[0].split("/")[-1], array1), (input_files[1].split("/")[-1], array2)], key=lambda x: len(x[1]), reverse=True)

        max_abs_sum = 0
        length_diff = len(array_large) - len(array_small)

        if length_diff>160000:
            return

        # for shift in range(-80000, length_diff+80001, 500):
        for shift in [0]:
            mixed_array = np.concatenate((np.zeros(max(-shift, 0)),
                                        array_large[max(0, shift):min(len(array_large), len(array_small) + shift)],
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

    audios = glob(path_pattern, recursive=True)
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
            pool.apply_async(func=mix_wav, kwds={"input_files":input_files,"output_file":output_file})

    pool.close()
    pool.join()
    print('结束', flush=True)

if __name__ == "__main__":

    print(os.getpid(), flush=True)
    # root = "/home/wangbin/workspaces/9_audio_qa_data/ASR_data/imda_speech/IMDA_raw/IMDA_-_National_Speech_Corpus_-_Additional"
    # # part6_audio_pattern = os.path.join(root, "PART6/Call_Centre_Design_*/Audio/**/*.wav")
    # # part5_audio_pattern = os.path.join(root, "PART5/*_Audio/**/*.wav")
    # part4_audio_pattern = os.path.join(root, "PART4/Codeswitching/Diff_Room_Audio/*.wav")
    # fire.Fire(mix_wavs(part4_audio_pattern, "PART4"))
    mix_wav(["/home/wangbin/workspaces/9_audio_qa_data/ASR_data/imda_speech/IMDA_raw/IMDA_-_National_Speech_Corpus_-_Additional/PART4/Codeswitching/Diff_Room_Audio/sur_0134_1268_phnd_cs-chn.wav",
                "/home/wangbin/workspaces/9_audio_qa_data/ASR_data/imda_speech/IMDA_raw/IMDA_-_National_Speech_Corpus_-_Additional/PART4/Codeswitching/Diff_Room_Audio/sur_0134_1269_phnd_cs-chn.wav"],
                "mixed_wav/PART4/diff_room/0134_chn.wav")
    print("complete", flush=True)


