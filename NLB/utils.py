from glob import glob
from collections import Counter
import os
import subprocess


audios=[os.path.basename(f)[:-4] for f in glob("/data/xunlong/NLB_data_preparation/data_NLB/train/audio/*", recursive=True)]
docx=[os.path.basename(f)[:-5] for f in glob("/data/xunlong/NLB_data_preparation/data_NLB/train/docx/*", recursive=True) if os.path.basename(f)[:-5] in audios]
files=[f for f in glob("/data/xunlong/NLB_data_preparation/data_NLB/train/transcripts/*/*", recursive=True) if os.path.basename(f)[:-5] in audios]


def move_docx(files):
    for f in files:
        filename=os.path.basename(f)
        os.rename(f, os.path.join("/data/xunlong/NLB_data_preparation/data_NLB/train/docx", filename))


def rename_files(files):
    files.sort(key=lambda x: len(os.path.basename(x)))
    for f in files:
        filename=os.path.basename(f)
        if len(filename)==18:
            audio_id=filename.split("OHC")[-1].split("_")[0]
            audio_seg=filename.split("OHC")[-1].split("_")[1]
            if audio_seg.startswith("0"):
                os.rename(f, os.path.join(os.path.dirname(f), audio_id+"-"+audio_seg[1:]))
            if audio_seg.startswith("1"):
                os.rename(f, os.path.join(os.path.dirname(f), audio_id+"-"+audio_seg))

        elif filename.startswith("00"):
            pass
        else:
            os.remove(f)


def convert_to_docx(files):
    for file in files:
        filename=os.path.basename(file)
        if not filename.endswith('.docx'):
            returncode = subprocess.call(
                ['soffice', '--headless', '--convert-to', 'docx:MS Word 2007 XML', '--outdir', os.path.dirname(file), file])

            if returncode != 0:
                print('failed convert {}'.format(file), flush=True)
                print('=========================================================', flush=True)
            else:
                print('success convert {}'.format(file), flush=True)
                os.remove(file)


files=glob("/mnt/home/zoux/datasets/NLB/train/docx/*")
rename_files(files)