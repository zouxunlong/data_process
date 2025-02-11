
import textgrid
from glob import glob
from tqdm import tqdm
import os

for part in ["PART3", "PART4", "PART5", "PART6"]:
    root           = "/home/users/astar/ares/zoux/scratch/workspaces/data_process/_data_in_processing/imda/imda_raw"
    textgrid_files = glob(f"{root}/{part}/Scripts/*.TextGrid")
    txt_files      = glob(f"{root}/{part}/txt_all/*.txt")
    os.makedirs(f"{root}/{part}/Scripts_fixed", exist_ok=True)

    for textgrid_file in tqdm(textgrid_files):
        id = textgrid_file.split("/")[-1].split(".")[0]
        if f"{root}/{part}/txt_all/{id}.txt" not in txt_files:
            print(f"{root}/{part}/txt_all/{id}.txt", flush=True)
            break

        if os.path.exists(textgrid_file.replace("/Scripts/", "/Scripts_fixed/")):
            print(f"{root}/{part}/txt_all/{id}.txt", flush=True)
            continue

        tg      = textgrid.TextGrid.fromFile(textgrid_file)
        minTime = tg.minTime
        maxTime = tg.maxTime
        tier    = tg.tiers[0]

        # Example modification: Change the start and end time of all intervals
        time_offset = float(open(f"{root}/{part}/txt_all/{id}.txt").readlines()[-1].split(" || ")[0].strip())
        time_offset = round(time_offset, 5)
        if abs(time_offset) < 1:
            continue

        for interval in tier:
            interval.minTime -= time_offset
            interval.minTime = max(min(interval.minTime, maxTime), minTime)
            interval.maxTime -= time_offset
            interval.maxTime = min(max(interval.maxTime, minTime), maxTime)

        # Save the modified TextGrid file
        tg.write(textgrid_file.replace("/Scripts/", "/Scripts_fixed/"))


