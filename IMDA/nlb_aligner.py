import fire
import stable_whisper
import json
import os
from tqdm import tqdm
from nltk.tokenize import sent_tokenize


def convert_to_txt(file):

    speaker_map = []
    with open(file, 'r') as f:
        for line in f:
            item = json.loads(line)
            speaker_map.extend([(item['speaker'], sent.strip())
                               for sent in sent_tokenize(item["utterance"])])

    speakers, sentences = zip(*speaker_map)
    return speakers, sentences


def force_align(audio, transcript_jsonl, model):

    speakers, sentences = convert_to_txt(transcript_jsonl)

    result = model.align(
        audio          = audio,
        text           = '\n'.join(sentences),
        language       = 'en',
        original_split = True
    )

    segments_timestamps = [{
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "segment_id": seg.id,
        "speaker": speakers[seg.id]
    } for seg in result.segments if seg.start != seg.end]

    return segments_timestamps


def main(gpu=0, start=0, end=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    model = stable_whisper.load_model('base')

    audio_dir = "/data/xunlong/NLB_data_preparation/data_NLB/test/audio"
    jsonl_dir = "/data/xunlong/NLB_data_preparation/data_NLB/test/jsonl"
    jsonl_aligned_dir = "/data/xunlong/NLB_data_preparation/data_NLB/test/jsonl_aligned"

    audio_files = os.listdir(audio_dir)
    audio_files.sort()

    for audio_file in tqdm(audio_files[start:end]):
        audio = f'{audio_dir}/{audio_file}'
        audio_id = audio_file.split('.')[0]
        transcript_jsonl = f'{jsonl_dir}/{audio_id}.jsonl'

        if os.path.exists(output_path_segments := f'{jsonl_aligned_dir}/{audio_id}.segments.jsonl'):
            continue

        segments_timestamps = force_align(audio, transcript_jsonl, model)

        with open(output_path_segments, 'w') as f:
            for segments_timestamp in segments_timestamps:
                f.write(json.dumps(segments_timestamp,
                        ensure_ascii=False) + '\n')


if __name__ == '__main__':
    fire.Fire(main)
