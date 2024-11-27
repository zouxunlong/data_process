import json
import os
import textgrid
import unicodedata
import re
from tqdm import tqdm


def normalize_sentence(sentence):
   if re.search('<(tamil|malay|mandarin)>', sentence):
      return ""
   sentence = unicodedata.normalize('NFKC', sentence)
   sentence = re.sub('<[a-zA-Z0-9/\s]*>', " ", sentence)
   sentence = re.sub('\((ppc|ppb|ppl|ppo)\)', " ", sentence, flags=re.IGNORECASE)
   sentence = " ".join(re.sub('(_|\(|\)|\[|\])', "", sentence).split()).strip()
   return sentence.strip()



root="/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw"
for part in ["PART3", "PART4", "PART5", "PART6"]:

   lines = open(f"{root}/{part}/erorr_files.jsonl").readlines()

   manifests = []
   manifests_with_transcriptions = []

   for line in tqdm(lines):
      item = json.loads(line)
      transcriptions = textgrid.TextGrid.fromFile((item["script_file"]))
      transcriptions = [{"start": interval.minTime, "end": interval.maxTime, "sentence": normalize_sentence(interval.mark)} for interval in transcriptions[0] if normalize_sentence(interval.mark)]
      text = " | ".join([interval["sentence"] for interval in transcriptions])
      if text:
         manifests.append({"audio_filepath": item["wav_file"], "text": text})
         manifests_with_transcriptions.append({"audio_filepath": item["wav_file"], "text": text, "transcriptions": transcriptions})
      
   file_to_filer=[]

   with open(f"{root}/{part}/manifest.jsonl", "w", encoding="utf-8") as f:
      for manifest in manifests:
         if os.path.basename(manifest["audio_filepath"]) in file_to_filer:
            continue
         f.write(json.dumps(manifest, ensure_ascii=False)+"\n")

   with open(f"{root}/{part}/manifest_with_transcriptions.jsonl", "w", encoding="utf-8") as f:
      for manifest in manifests_with_transcriptions:
         if os.path.basename(manifest["audio_filepath"]) in file_to_filer:
            continue
         f.write(json.dumps(manifest, ensure_ascii=False)+"\n")

   print(f"Done {part}", flush=True)
