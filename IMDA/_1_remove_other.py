import json
import os
from tqdm import tqdm



root="/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw"
for part in ["PART3", "PART4", "PART5", "PART6"]:

   lines = open(f"{root}/{part}/erorr_files.jsonl").readlines()
   wav_filenames=[]
   for line in tqdm(lines):
      item = json.loads(line)
      wav_filenames.append(os.path.basename(item["wav_file"]).split(".")[0])

   all_files=os.listdir(f"{root}/{part}/NFA_output/ctm/segments")
   files_to_remove=[file for file in all_files if not file.split(".")[0] in wav_filenames]

   for file in files_to_remove:
      os.remove(f"{root}/{part}/NFA_output/ctm/segments/{file}")
   print(f"Done {part}", flush=True)