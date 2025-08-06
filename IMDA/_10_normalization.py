from nemo.collections.nlp.models import PunctuationCapitalizationModel
from datasets import load_from_disk, Value
import re
import string
import warnings
import fire
import torch
from multiprocess import set_start_method
from glob import glob
import os

warnings.filterwarnings('ignore')

model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")

class Reg_Exp:
    pattern_punctuation = r"""[!?,*.:;"#$£€%&'()+-/<≤=≠≥>@[\]^_{|}，。、—‘’“”：；【】￥…《》？！（）]"""
    pattern_url = r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    pattern_email = r"[\w\-\.]+@([\w\-]+\.)+[\w\-]{2,4}"
    pattern_arabic = r"[\u0600-\u06FF]"
    pattern_chinese = r"[\u4e00-\u9fff]"
    pattern_tamil = r"[\u0B80-\u0BFF]"
    pattern_thai = r"[\u0E00-\u0E7F]"
    pattern_russian = r"[\u0400-\u04FF]"
    pattern_korean = r"[\uac00-\ud7a3]"
    pattern_japanese = r"[\u3040-\u30ff\u31f0-\u31ff]"
    pattern_vietnamese = r"[àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]"
    pattern_emoji = r'[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F64F\U0001F680-\U0001FAFF\U00002702-\U000027B0]'


def check_same(sentences, predictions):

    need_further_process = []
    for i in range(len(sentences)):
        old_clean_punc = sentences[i].translate(str.maketrans("", "", string.punctuation)).replace('-', ' ')
        old_clean_punc = re.sub(Reg_Exp.pattern_punctuation, ' ', old_clean_punc)
        old_clean_punc = " ".join(old_clean_punc.split()).strip()

        new_clean_punc = predictions[i].translate(str.maketrans("", "", string.punctuation)).replace('-', ' ')
        new_clean_punc = re.sub(Reg_Exp.pattern_punctuation, ' ', new_clean_punc)
        new_clean_punc = " ".join(new_clean_punc.split()).strip()

        if old_clean_punc.lower() == new_clean_punc.lower():
            need_further_process.append(False)
        else:
            need_further_process.append(True)
    return need_further_process


def norm_map(batch):

    sentences            = [other_attributes['original_answer'] for other_attributes in batch['other_attributes']]
    need_further_process = [other_attributes['need_further_process'] for other_attributes in batch['other_attributes']]
    
    if True in need_further_process:
        try:
            predictions = model.add_punctuation_capitalization(sentences)
            predictions = [text.replace("<speaker2>", "<Speaker2>").replace("<speaker1>", "<Speaker1>") for text in predictions]
            need_further_process = check_same(sentences, predictions)
            for i in range(len(sentences)):
                batch['answer'][i]['text']                           = predictions[i]
                batch['other_attributes'][i]['original_answer']      = sentences[i]
                batch['other_attributes'][i]['need_further_process'] = need_further_process[i]
                batch['other_attributes'][i]['normalization_tool']   = "nemo"
            return batch

        except:
            predictions = sentences
            need_further_process = [True]*len(predictions)
            for i in range(len(sentences)):
                batch['answer'][i]['text']                           = predictions[i]
                batch['other_attributes'][i]['original_answer']      = sentences[i]
                batch['other_attributes'][i]['need_further_process'] = need_further_process[i]
                batch['other_attributes'][i]['normalization_tool']   = "nemo"
            return batch
    
    else:
        return batch




def main():

    ds_paths = glob(f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/from_yx/imda/imda_asr/train/ASR_normalized/*")

    for ds_path in sorted(ds_paths, reverse=False):
        if os.path.exists(ds_path.replace("/ASR_normalized/", "/ASR_normalized2/")):
            continue
        ds                                                   = load_from_disk(ds_path)
        features                                             = ds.features
        features['other_attributes']['original_answer']      = Value('string')
        features['other_attributes']['need_further_process'] = Value('bool')
        features['other_attributes']['normalization_tool']   = Value('string')

        print(ds, flush=True)

        processed_ds = ds.map(norm_map,
                              batched           = True,
                              batch_size        = 1,
                              features          = features,
                              writer_batch_size = 1,
                              num_proc          = 1,
                              )
        output_folder = ds_path.replace("/ASR_normalized/", "/ASR_normalized2/")
        processed_ds.save_to_disk(output_folder, num_proc=10)


if __name__ == "__main__":
    fire.Fire(main)
