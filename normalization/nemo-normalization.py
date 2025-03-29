from nemo.collections.nlp.models import PunctuationCapitalizationModel

import os
from datasets import load_from_disk
from glob import glob
import re
import string

import warnings 

warnings.filterwarnings('ignore')

# Download and load the pre-trained BERT-based model
model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")




def check_same(transctiption_lists, prediction_lists):
    # clean space and punc
    need_further_proecess_list = []
    for i in range(len(transctiption_lists)):
        old_clean_space = re.sub(r"\s\s+", ' ', transctiption_lists[i]).strip()
        old_clean_punc = old_clean_space.translate(str.maketrans("","", string.punctuation)).replace('\n', ' ')
        old_clean_punc = re.sub(r"\s\s+", ' ', old_clean_punc).strip()

        new_clean_space = re.sub(r"\s\s+", ' ', prediction_lists[i]).strip()
        new_clean_punc = new_clean_space.translate(str.maketrans("","", string.punctuation)).replace('-', ' ')
        new_clean_punc = re.sub(r"\s\s+", ' ', new_clean_punc).strip()

        if old_clean_punc.lower() == new_clean_punc.lower():
            need_further_proecess_list.append(False)
        else:
            need_further_proecess_list.append(True)
    return need_further_proecess_list

def norm_map(batch):

    # try:
    sentences = [answer["text"] for answer in batch['answer']]
    predictions = model.add_punctuation_capitalization(sentences)


    for i in range(len(sentences)):
        batch['answer'][i]['text'] = predictions[i]
        batch['other_attributes'][i]['need_further_proecess'] = need_further_proecess_list[i]

    # except:
    #     breakpoint()
    
    return batch

for data_path in glob(f"/data/projects/13003558/zoux/datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/gigaspeech_30_ASR"):
        
    name = os.path.basename(data_path)
    output_folder = "/data/projects/13003558/zoux/gigaspeech_30_ASR_normalized"

    ds = load_from_disk(data_path)

    processed_ds = ds.map(norm_map,
                          batched = True,
                          batch_size= 512*2,
                          writer_batch_size = 1,
                          num_proc=1,
                          )

    processed_ds.save_to_disk(output_folder)



    