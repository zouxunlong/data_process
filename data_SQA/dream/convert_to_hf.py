from datasets import Dataset, Audio

audio_dataset = Dataset.from_json("/home/user/data/data_SQA/dream/dream_tts_json/dream.test.jsonl", split="test").cast_column("audio", Audio(sampling_rate=16000))

audio_dataset.save_to_disk("/home/user/data/data_SQA/dream/dream_tts_hf/test")


# import librosa, json
# def get_duration_librosa(file_path):
#    audio_data, sample_rate = librosa.load(file_path)
#    duration = librosa.get_duration(y=audio_data, sr=sample_rate)
#    return duration

# with open("dream.test.jsonl") as f_in, open("dream.test.lessThan30.jsonl", "w", encoding="utf8") as f_out:
#    for line in f_in:
#         item=json.loads(line)
#         audio_length=get_duration_librosa(item["audio"])
#         if audio_length<30:
#             f_out.write(line)
