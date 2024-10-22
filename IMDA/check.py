 
from collections import defaultdict
from copy import copy
import gc
import json
import random
from datasets import load_from_disk, Audio, Features, Value
import os

print(os.getpid(), flush=True)
 
instructions_asr = [
    "Convert the spoken words to written text.",
    "Transcribe the verbal content into a document.",
    "Please create a text transcription of the audio.",
    "Kindly convert the audio into text.",
    "Please write down the contents of the audio.",
    "Please document the spoken words as text.",
    "Turn the spoken language into written form.",
    "Produce a text version of the spoken content.",
    "Transcribe the audio recording into text.",
    "Document the contents of this audio in written form.",
    "Capture the audio in written text.",
    "Render the spoken audio into written words.",
    "Convert the audio file to a textual document.",
    "Provide a written transcription of the audio.",
    "Translate the spoken words into text format.",
    "Record the audio content as written text.",
    "Write out the audio into a text file.",
    "Generate a text document from this audio.",
    "Transform the audio speech into text.",
    "Create a written record of the audio.",
    "Convert these spoken words into a text format.",
    "Please produce a text copy of this sound recording.",
    "Transcribe this audio into written words.",
    "Make a textual transcription of the spoken audio.",
    "Document the verbal audio into text.",]


def segment_batch(batch):
    new_batch = defaultdict(list)

    for i in range(len(batch["audio"])):
        if batch["transcription1"][i]:
            array_chunk=batch["audio"][i]["array"][:2]
            new_array=array_chunk.copy()
            new_batch["context"].append({
                        "text": None,
                        "audio": {"array": new_array, "sampling_rate": batch["audio"][i]["sampling_rate"]}
                    })
            # for transcription in batch["transcription1"][i]:
            #     start = transcription["start"]
            #     end = transcription["end"]
            #     sentence = transcription["sentence"]
            #     segment_array = batch["audio"][i]["array"][max(
            #         0, int(0*16000)):int(2)]
            #     if segment_array.size > 0:
            #         new_batch["context"].append({
            #             "text": None,
            #             "audio": {"array": segment_array, "sampling_rate": batch["audio"][i]["sampling_rate"]}
            #         })
            #         new_batch["instruction"].append({
            #             "text": random.choice(instructions_asr),
            #             "audio": None
            #         })
            #         new_batch["answer"].append({
            #             "text": sentence,
            #             "audio": None
            #         })
            #         new_batch["other_attributes"].append({
            #             "conversation_id": batch["conversation_id"][i],
            #             "settings": batch["settings"][i],
            #             "partition": batch["partition"][i],
            #             "speaker": batch["speaker1"][i],
            #         })
            #         break
        else:
            print("empty transcription1 at {}--------------------".format(batch["conversation_id"][i]), flush=True)

        # if batch["transcription2"][i]:
        #     for transcription in batch["transcription2"][i]:
        #         start = transcription["start"]
        #         end = transcription["end"]
        #         sentence = transcription["sentence"]
        #         segment_array = batch["audio"][i]["array"][max(0, int(start*16000)):int(end*16000)]
        #         if segment_array.size > 0:
        #             new_batch["context"].append({
        #                 "text": None,
        #                 "audio": {"array": segment_array, "sampling_rate": batch["audio"][i]["sampling_rate"]}
        #             })
        #             new_batch["instruction"].append({
        #                 "text": random.choice(instructions_asr),
        #                 "audio": None
        #             })
        #             new_batch["answer"].append({
        #                 "text": sentence,
        #                 "audio": None
        #             })
        #             new_batch["other_attributes"].append({
        #                 "conversation_id": batch["conversation_id"][i],
        #                 "settings": batch["settings"][i],
        #                 "partition": batch["partition"][i],
        #                 "speaker": batch["speaker2"][i],
        #             })
        #             break
        # else:
        #     print("empty transcription2 at {}--------------------".format(batch["conversation_id"][i]), flush=True)

    # del batch
    # gc.collect()
    
    if len(new_batch["context"])==0:
        print("empty new_batch at {}--------------------".format(batch["conversation_id"][i]), flush=True)
        return {
            'context': [None],
            # 'instruction': [None],
            # 'answer': [None],
            # 'other_attributes': [None],
        }
    else:
        return new_batch


def filter(example):
    if example["context"]:
        return True
    return False

 
ds = load_from_disk("PART4.hf/train")

features = Features({
        'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        # 'instruction': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        # 'answer': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        # 'other_attributes': {
        #     "conversation_id": ds.features["conversation_id"],
        #     "settings": ds.features["settings"],
        #     "partition": ds.features["partition"],
        #     "speaker": ds.features["speaker1"],
        # }
    })

audio_dataset = ds.map(
    segment_batch,
    features             = features,
    num_proc             = 20,
    batched              = True,
    batch_size           = 1,
    writer_batch_size    = 1,
    remove_columns       = ds.column_names,
)
 
dataset=audio_dataset.filter(filter, num_proc=10)
audio_dataset.save_to_disk("PART4.ASR.schemed/train", num_proc=10)

