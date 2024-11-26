from pydub import AudioSegment
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

files=glob("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART*/*/*.wav")

files.sort()

    
def convert_to_mono(file):
    # Load the audio file
    audio = AudioSegment.from_wav(file)

    # Check the number of channels
    if audio.channels > 1:
        # Convert to mono
        audio = audio.set_channels(1)
        # Export the mono audio file
        audio.export(file, format="wav")
        print(f"Converted to mono and saved as {file}")

    
def convert_samplingrate_to_16k(file):
    # Load the audio file
    audio = AudioSegment.from_wav(file)
    # Check the sampling rate
    if audio.frame_rate != 16000:
        # Convert to 16kHz
        audio = audio.set_frame_rate(16000)
        # Export the 16kHz audio file
        audio.export(file, format="wav")
        print(f"Converted to 16kHz and saved as {file}", flush=True)

with Pool(112) as p:
    results = list(tqdm(p.imap(convert_samplingrate_to_16k, files), total=len(files)))
    