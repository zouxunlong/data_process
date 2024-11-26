import json
import os
from fire import Fire
import logging
from glob import glob
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate
import textgrid
import stable_whisper


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

metric     = evaluate.load("wer")
normalizer = BasicTextNormalizer()
processor  = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="English", task="transcribe")
model      = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model.to(device)

model_align = stable_whisper.load_model('base')

def calculate_slots_wer(script_file, wav_file):

    array, sampling_rate = sf.read(wav_file)
    transcriptions = textgrid.TextGrid.fromFile(script_file)
    transcriptions = [(interval.minTime, interval.maxTime, interval.mark) for interval in transcriptions[0] if len(interval.mark.split()) >= 5 and interval.maxTime-interval.minTime < 30]
    
    starts, ends, labels=zip(*transcriptions)

    chunk_arrays = [array[int(starts[i]*16000):int(ends[i]*16000)] for i in range(len(starts))]
    
    batch_size = 20
    wers = []

    for i in range(0, len(chunk_arrays), batch_size):
        batch_chunks = chunk_arrays[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        input_features = processor.feature_extractor(batch_chunks, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
        generated_ids = model.generate(inputs=input_features, language='en')
        batch_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

        batch_preds = [normalizer(pred) for pred in batch_preds]
        batch_labels = [normalizer(label) for label in batch_labels]

        for j in range(len(batch_preds)):
            wer = 100 * metric.compute(predictions=[batch_preds[j]], references=[batch_labels[j]])
            wers.append((starts[i+j], ends[i+j], wer, batch_preds[j], batch_labels[j]))
    print(wers, flush=True)
    return wers


def force_align(audio, sentence, model_align):
    
    result = model_align.align(
        audio          = audio,
        text           = '\n'.join([sentence]),
        language       = 'en',
        original_split = True
    )

    segments_timestamps = [{
        "start"     : seg.start,
        "end"       : seg.end,
        "text"      : seg.text,
        "segment_id": seg.id,
        "speaker"   : speakers[seg.id]
    } for seg in result.segments if seg.start!=seg.end]

    return segments_timestamps

def calculate_actual_time(script_file, wav_file):
    return 0

def try_fix(file):
    lines=open(file).readlines()
    for line in lines:
        item         = json.loads(line)
        script_file  = item["script_file"]
        wav_file     = item["wav_file"]
        wers         = calculate_slots_wer(script_file, wav_file)
        item["wers"] = wers
        breakpoint()


def main():
    for file in sorted(glob("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART*/erorr_files.jsonl")):
        try_fix(file)


if __name__ == "__main__":
    logging.error(f"Script executed starts")
    Fire(main)
    logging.error(f"Script executed ends")
