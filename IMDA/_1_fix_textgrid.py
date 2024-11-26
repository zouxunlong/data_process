import json
import os
from fire import Fire
import logging
from glob import glob
import numpy as np
import soundfile as sf
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import textgrid
import stable_whisper


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

os.environ["CUDA_VISIBLE_DEVICES"] = str("0,1,2,3,4,5,6,7")
model_align = stable_whisper.load_model('base')


def force_align(script_file, wav_file):

    array, sampling_rate = sf.read(wav_file)
    transcriptions = textgrid.TextGrid.fromFile(script_file)
    transcriptions = [(interval.minTime, interval.maxTime, interval.mark, array[int(max(interval.minTime-5, 0)*16000):int((interval.maxTime+5)*16000)]) for interval in transcriptions[0] if len(interval.mark.split()) >= 5 and all(c.isalpha() or c.isspace() for c in interval.mark)]
    
    for start, end, label, chunk_array in transcriptions:

        print(chunk_array.dtype, flush=True)
        chunk_array = chunk_array.astype(np.float32)
        print(chunk_array.dtype, flush=True)

        result = model_align.align(
            audio          = chunk_array,
            text           = label,
            language       = 'en',
            original_split = True
        )

        segments_timestamps = [{
            "start"     : seg.start,
            "end"       : seg.end,
            "text"      : seg.text,
        } for seg in result.segments if seg.start!=seg.end]
        transcription={"start":start, "end":end, "label":label}
        breakpoint()

    return segments_timestamps


def try_fix(file):
    lines=open(file).readlines()
    for line in lines:
        item         = json.loads(line)
        script_file  = item["script_file"]
        wav_file     = item["wav_file"]
        segments_timestamps         = force_align(script_file, wav_file)
        item["segments_timestamps"] = segments_timestamps
        breakpoint()


def main():
    for file in sorted(glob("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART*/erorr_files.jsonl")):
        try_fix(file)


if __name__ == "__main__":
    logging.error(f"Script executed starts")
    Fire(main)
    logging.error(f"Script executed ends")

