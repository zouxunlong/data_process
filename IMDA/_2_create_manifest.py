import json
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
   return sentence

root="/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw"
for part in ["PART3", "PART4", "PART5", "PART6"]:

   lines = open(f"{root}/{part}/wav_script_pairs.jsonl").readlines()

   manifests = []

   for line in tqdm(lines):
      item = json.loads(line)
      transcriptions = textgrid.TextGrid.fromFile((item["script_file"]))
      transcriptions = [{"start": interval.minTime, "end": interval.maxTime, "sentence": normalize_sentence(interval.mark)} for interval in transcriptions[0] if len(normalize_sentence(interval.mark).split())>1]
      text = " | ".join([interval["sentence"] for interval in transcriptions])
      if text:
         manifests.append({"audio_filepath": item["wav_file"], 
                           "text": text,
                           "transcriptions": transcriptions})
   
   batch_size=len(manifests)//8
   open(f"{root}/{part}/manifest.jsonl", "w", encoding="utf-8").write("\n".join([json.dumps(manifest, ensure_ascii=False) for manifest in manifests]))

   open(f"{root}/{part}/manifest0.jsonl", "w", encoding="utf-8").write("\n".join([json.dumps(manifest, ensure_ascii=False) for manifest in manifests[:batch_size]]))
   open(f"{root}/{part}/manifest1.jsonl", "w", encoding="utf-8").write("\n".join([json.dumps(manifest, ensure_ascii=False) for manifest in manifests[batch_size:2*batch_size]]))
   open(f"{root}/{part}/manifest2.jsonl", "w", encoding="utf-8").write("\n".join([json.dumps(manifest, ensure_ascii=False) for manifest in manifests[2*batch_size:3*batch_size]]))
   open(f"{root}/{part}/manifest3.jsonl", "w", encoding="utf-8").write("\n".join([json.dumps(manifest, ensure_ascii=False) for manifest in manifests[3*batch_size:4*batch_size]]))
   open(f"{root}/{part}/manifest4.jsonl", "w", encoding="utf-8").write("\n".join([json.dumps(manifest, ensure_ascii=False) for manifest in manifests[4*batch_size:5*batch_size]]))
   open(f"{root}/{part}/manifest5.jsonl", "w", encoding="utf-8").write("\n".join([json.dumps(manifest, ensure_ascii=False) for manifest in manifests[5*batch_size:6*batch_size]]))
   open(f"{root}/{part}/manifest6.jsonl", "w", encoding="utf-8").write("\n".join([json.dumps(manifest, ensure_ascii=False) for manifest in manifests[6*batch_size:7*batch_size]]))
   open(f"{root}/{part}/manifest7.jsonl", "w", encoding="utf-8").write("\n".join([json.dumps(manifest, ensure_ascii=False) for manifest in manifests[7*batch_size:]]))
   
   

         
         

   print(f"Done {part}", flush=True)
