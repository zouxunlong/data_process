from glob import glob
import json
from datasets import Dataset, Audio, Features, Value
import fire


features = Features({
    "audio": Audio(sampling_rate=16000, decode=True),
    'transcriptions': [{
        "start"     : Value(dtype='float32'),
        "end"       : Value(dtype='float32'),
        "speaker"   : Value(dtype='string'),
        "text"      : Value(dtype='string'),
        "segment_id": Value(dtype='int32')
    }]
})

def map_fn(example, jsonl_aligned_dir):

    audio_id = example['audio'].split('/')[-1].split('.')[0]
    filename = jsonl_aligned_dir + audio_id + '.segments.jsonl'

    items = [json.loads(line) for line in open(filename)]
    segments_split_time = [items[0]['start']] + [(items[i]['end'] + items[i+1]['start'])/2 for i in range(len(items)-1)] + [items[-1]['end']]
    for i, item in enumerate(items):
        item["start"] = segments_split_time[i]
        item["end"]   = segments_split_time[i+1]

    example['transcriptions']=items

    return example


def build_hf(
    audio_dir         = "/data/xunlong/NLB_data_preparation/data_NLB/test/audio",
    jsonl_aligned_dir = "/data/xunlong/NLB_data_preparation/data_NLB/test/jsonl_aligned/",
    des_path          = "/data/xunlong/NLB_data_preparation/data_NLB/test/NLB_v1",
):

    audio_files = glob(f"{audio_dir}/*.mp3")
    audio_files.sort()
    audio_dataset = Dataset.from_dict({"audio": audio_files})

    ds = audio_dataset.map(
        function          = map_fn,
        fn_kwargs         = {'jsonl_aligned_dir': jsonl_aligned_dir},
        features          = features,
        num_proc          = 10,
        writer_batch_size = 1,
    )
    ds.save_to_disk(des_path, num_proc=4)


if __name__ == "__main__":
    fire.Fire(build_hf)
