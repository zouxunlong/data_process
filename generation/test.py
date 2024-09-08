from datasets import load_from_disk
import tempfile
import numpy
import soundfile as sf

def convert(example):
    audio_array=example["context"]["audio"]["array"]
    fname=tempfile.NamedTemporaryFile(dir="temp", suffix=".mp3").name
    # audio_array=numpy.concatenate([audio_array]*10)
    sf.write(fname, audio_array, 16000)
    example["context"]["audio"]={"bytes": open(fname, "rb").read()}
    return example

split="/mnt/data/all_datasets/datasets_multimodal/other_prepared/ASR/IMDA_PART6_300_ASR_v2/test"
ds=load_from_disk(split)
ds2=ds.map(convert, num_proc=32, batch_size=1, writer_batch_size=1, features=ds.features)
ds2.save_to_disk(split.replace("datasets_multimodal", "datasets_multimodal_new"), num_proc=4)

