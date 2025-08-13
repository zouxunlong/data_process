from glob import glob
import io
from datasets import Dataset, Value, Audio
import librosa
from tqdm import tqdm
import os
from multiprocessing import Pool
import tempfile
import chardet
import soundfile as sf
from itertools import groupby
import re


data_dir   = "/mnt/data/all_datasets/data_process/_data_in_processing/500Hours_MinnanDialectConversationalSpeechDataByMobilePhone"
output_dir = "/mnt/data/all_datasets/data_process/_data_in_processing/500Hours_MinnanDialectConversationalSpeechDataByMobilePhone_ASR"


class ASRDataset(object):
    def __init__(self, data_dir, process_workers=16, save_workers=1):
        self.data_dir        = data_dir
        self.process_workers = process_workers
        self.save_workers    = save_workers

    def get_ogg_bytes(self, audio_array, samplerate, start_time=None, end_time=None):
        if start_time is not None and end_time is not None:
            audio_array = audio_array[int(start_time * samplerate) : int(end_time * samplerate)]
        audio_length = audio_array.shape[0] / samplerate
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, 16000, format="OGG", subtype="OPUS")
        return audio_length, buffer.getvalue()

    def parse_metadata(self, metadata_path):
        """Parse app.metadata file to extract speaker and recording information."""
        metadata = {}
        with open(metadata_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('CMT'):
                continue
            
            parts = line.split('\t', 1)
            if len(parts) == 2:
                key, value = parts
                metadata[key] = value
        
        return metadata

    def extract_speaker_info(self, metadata):
        """Extract speaker information from metadata."""
        speakers = {}
        
        # Parse speaker codes
        if 'SCD' in metadata:
            speaker_codes = metadata['SCD'].split(',')
        else:
            speaker_codes = []
        
        # Parse gender info
        gender_info = {}
        if 'SEX' in metadata:
            for item in metadata['SEX'].split(','):
                if '_' in item:
                    speaker, gender = item.split('_', 1)
                    gender_info[speaker] = gender
        
        # Parse age info
        age_info = {}
        if 'AGE' in metadata:
            for item in metadata['AGE'].split(','):
                if '_' in item:
                    speaker, age = item.split('_', 1)
                    age_info[speaker] = age
        
        # Parse accent info
        accent_info = {}
        if 'ACC' in metadata:
            for item in metadata['ACC'].split(','):
                if '_' in item:
                    speaker, accent = item.split('_', 1)
                    accent_info[speaker] = accent

        for speaker_code in speaker_codes:
            speaker_data = {
                'speaker_id': speaker_code,
                'gender'    : gender_info.get(speaker_code, 'unknown'),
                'age'       : age_info.get(speaker_code, 'unknown'),
                'accent'    : accent_info.get(speaker_code, 'unknown')
            }
            speakers[speaker_code] = speaker_data
        
        return speakers

    def get_entries(self):

        metadata_files = sorted(glob(f"{data_dir}/**/*.metadata", recursive=True))

        with Pool(processes=self.process_workers) as pool:

            data_entries = []
            for metadata_file in tqdm(metadata_files, desc="Processing metadata files"):
                txt_file = metadata_file.replace(".metadata", ".txt")
                wav_file = metadata_file.replace(".metadata", ".wav")
                if not os.path.exists(txt_file) or not os.path.exists(wav_file):
                    continue
                metadata = self.parse_metadata(metadata_file)
                speakers = self.extract_speaker_info(metadata)

                with open(txt_file, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()

                entry = {
                    'wav_file'           : wav_file,
                    'transcription'      : transcription,
                    'recording_condition': metadata.get('SCC', ''),
                    'microphone_distance': metadata.get('MIP', ''),
                    'device_info'        : metadata.get('MIT', ''),
                    'category'           : metadata.get('CCD', ''),
                    'environment'        : metadata.get('REP', ''),
                    'speakers'           : speakers,
                    'num_speakers'       : len(speakers),
                    'dialect'            : 'minnan_hokken'
                }
                data_entries.append(entry)
        return data_entries

    def entry2samples(self, entry):
        """Convert dataset entry to samples."""
        samples = []

        audio_array, samplerate = sf.read(entry['wav_file'])
        if len(audio_array.shape) > 1:
            audio_array = audio_array.sum(axis=1)
        if samplerate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=samplerate, target_sr=16000)
            samplerate = 16000
        
        for trnascription in entry['transcription'].split('\n'):
            items = trnascription.split()
            if len(items)==4:
                start, end, speaker_id, text = items
                if text.strip() == "":
                    continue
                audio_length, bytes = self.get_ogg_bytes(audio_array, samplerate, start_time=float(start), end_time=float(end))
                sample = {
                    'context'         : {'text': None, 'audio': {'bytes': bytes}},
                    'instruction'     : {'text': 'Transcribe.', 'audio': None},
                    'answer'          : {'text': text.strip(), 'audio': None},
                    'other_attributes': {
                        'speaker'            : entry['speakers'][speaker_id],
                        'recording_condition': entry['recording_condition'],
                        'microphone_distance': entry['microphone_distance'],
                        'device_info'        : entry['device_info'],
                        'category'           : entry['category'],
                        'environment'        : entry['environment'],
                        'dialect'            : entry['dialect']
                    },
                    'audio_length': audio_length,
                    'language'    : 'hokken',
                }
                samples.append(sample)
        return samples


    def create_dataset(self):

        data_entries = self.get_entries()
        with Pool(processes=16) as pool:
            examples_list = list(tqdm(pool.imap_unordered(self.entry2samples, data_entries), total=len(data_entries)))
        print(f"Total examples_list: {len(examples_list)}")
        data_samples = [example for examples in examples_list for example in examples]
        print(f"Total data_samples: {len(data_samples)}")

        ds = Dataset.from_list(data_samples)
        ds = ds.cast_column("instruction", {'text': Value(dtype='string', id=None), 'audio': Audio(sampling_rate=16000, decode=True)})
        ds = ds.cast_column("answer", {'text': Value(dtype='string', id=None), 'audio': Audio(sampling_rate=16000, decode=True)})
        ds = ds.cast_column("context", {'text': Value(dtype='string', id=None), 'audio': Audio(sampling_rate=16000, decode=True)})
        ds.save_to_disk(str(output_dir), num_proc=self.save_workers)

asr_d = ASRDataset(data_dir)
ds = asr_d.create_dataset()

