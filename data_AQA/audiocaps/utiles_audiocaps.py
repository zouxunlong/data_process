from datasets import load_from_disk
# import soundfile as sf
# import os


def shorten_array_with_wav_generation(example):

    # global wav_ids

    # youtube_id=example["youtube_id"]
    array=example["audio"]["array"]
    sr=example["audio"]["sampling_rate"]
    st=example["start_time"]
    example["audio"]["array"]=array[sr*st:sr*(st+10)]

    # if youtube_id in wav_ids:
    #     print("{}.wav exists.--------------------------".format(youtube_id), flush=True)

    # elif not array[sr*st:sr*(st+10)].any():
    #     print("{}.wav is empty.--------------------------".format(youtube_id), flush=True)

    # else:
    #     sf.write('audiocaps_audio/{}.wav'.format(youtube_id), array[sr*st:sr*(st+10)], sr)
    #     print("{}.wav created".format(youtube_id), flush=True)
    #     wav_ids.add(youtube_id)

    return example


def shorten_array(example):
    array=example["audio"]["array"]
    sr=example["audio"]["sampling_rate"]
    st=example["start_time"]
    example["audio"]["array"]=array[sr*st:sr*(st+10)]
    return example


def filter(example):
    array=example["audio"]["array"]
    if array.any():
        return True
    else:
        return False

if __name__ == "__main__":

    dataset_path = "/home/user/data/data_AQA/audiocaps/audiocaps_hf/train"
    print("loading..", flush=True)
    dataset = load_from_disk(dataset_path)
    print("loaded..", flush=True)
    print("filtering...", flush=True)
    updated_dataset = dataset.filter(filter)
    print("filtered...", flush=True)
    print("saving...", flush=True)
    updated_dataset.save_to_disk("{}_filtered".format(dataset_path))
    print("completed", flush=True)

