import torch
from TTS.api import TTS
import json, os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

with open('dialogues_final.jsonl', 'r') as f:
    data=[]
    for line in f:
        data.append(json.loads(line))


for sample in tqdm(data):

    index = sample['dialogue_id']
    dialogue  = sample['dialogue']

    # skip if already exists
    if os.path.exists('tts_audio/{}'.format(index)):
        print('-----\n tts_audio/{} already exists\n-----'.format(index))
        continue
    else:
        os.makedirs('tts_audio/{}'.format(index))
        print("Directory tts_audio/{} is created!".format(index))
        for i, (speaker, utterance) in enumerate(dialogue):
            tts.tts_to_file(text=utterance, speaker=speaker, language="en", file_path="tts_audio/{}/{}.wav".format(index, i))

