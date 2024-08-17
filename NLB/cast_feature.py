from datasets import load_from_disk, Features, Value, Audio

# features = Features({
#     'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
#     'instruction': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
#     'answer': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
#     'other_attributes': {
#         "audio_path": Value(dtype='string'),
#         "duration": Value(dtype='string'),
#         "start_time": Value(dtype='float32'),
#         "end_time": Value(dtype='float32'),
#     }
# })

features=Features({'audio': Audio(sampling_rate=16000, mono=True, decode=True, id=None), 'transcriptions': [{'start': Value(dtype='float32', id=None), 'end': Value(dtype='float32', id=None), 'speaker': Value(dtype='string', id=None), 'text': Value(dtype='string', id=None), 'segment_id': Value(dtype='int32', id=None)}]})

ds=load_from_disk('/mnt/home/zoux/datasets/NLB/test/NLB_v1').cast(features=features, num_proc=160)
ds.save_to_disk('/mnt/home/zoux/datasets/NLB/test/NLB_v2', num_proc=16)

print("complete", flush=True)
