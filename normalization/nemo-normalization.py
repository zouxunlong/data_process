from nemo.collections.nlp.models import PunctuationCapitalizationModel

import os
from datasets import load_from_disk
from glob import glob
import re
import string

import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore')

# Download and load the pre-trained BERT-based model
model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")

# sentences = ["""in 2018 cornell researchers built a high-powered detector that in combination with an algorithm-driven process called ptychography set a world record 
#             by tripling the resolution of a state-of-the-art electron microscope as successful as it was that approach had a weakness it only worked with ultrathin samples that were
#             a few atoms thick anything thicker would cause the electrons to scatter in ways that could not be disentangled now a team again led by david muller the samuel b eckert
#             professor of engineering has bested its own record by a factor of two with an electron microscope pixel array detector empad that incorporates even more sophisticated
#             3d reconstruction algorithms the resolution is so fine-tuned the only blurring that remains is the thermal jiggling of the atoms themselves""" ] * 1000

# for i in range(10000):

#     result = model.add_punctuation_capitalization(sentences)

# breakpoint()

# train_folder = "/home/users/astar/ares/liuz4/scratch/data_processing/ASR_normalized_opus/train/ASR"
train_folder = "/home/users/astar/ares/liuz4/scratch/data_processing/ASR_normalized_opus/output/round1/train"
train_save_folder = "/home/users/astar/ares/liuz4/scratch/data_processing/ASR_normalized_opus/output/nemo"

if not os.path.exists(train_save_folder):
    os.makedirs(train_save_folder)
    print(f"Create Saving Folder for Train")


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
    sentences = [batch['other_attributes'][i]['transcription'] for i in range(len(batch['other_attributes']))]
    predictions = model.add_punctuation_capitalization(sentences)

    # breakpoint()

    need_further_proecess_list = check_same(sentences, predictions)

    for i in range(len(sentences)):
        batch['answer'][i]['text'] = predictions[i]
        batch['other_attributes'][i]['need_further_proecess'] = need_further_proecess_list[i]

    # except:
    #     breakpoint()
    
    return batch

for data_path in glob(f"{train_folder}/*"):
        
    name = os.path.basename(data_path)
    folder_name = os.path.basename(data_path)
    if folder_name =='AIShell_zh_ASR_v2':
        continue
    output_folder = os.path.join(train_save_folder, folder_name)
    if os.path.exists(output_folder):
        continue
    
    if 'IMDA' in folder_name:
        continue

    print(f"Process {folder_name}")
    ds = load_from_disk(data_path)


    processed_ds = ds.map(norm_map,
                          batched = True,
                          batch_size= 512*2,
                          writer_batch_size = 1,
                          num_proc=1,
                          # load_from_cache_file=False
                          )
    
    processed_ds.save_to_disk(output_folder)

    # breakpoint()


    