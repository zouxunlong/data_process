from nemo.collections.nlp.models import PunctuationCapitalizationModel
from datasets import load_from_disk, Value
import re
import string
import warnings 
import fire
import torch
from multiprocess import set_start_method

warnings.filterwarnings('ignore')

model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")


def check_same(sentences, predictions):

    need_further_proecess_list = []
    for i in range(len(sentences)):
        old_clean_space = re.sub(r"\s\s+", ' ', sentences[i]).strip()
        old_clean_punc = old_clean_space.translate(str.maketrans("","", string.punctuation)).replace('\n', ' ')
        old_clean_punc = re.sub(r"\s\s+", ' ', old_clean_punc).strip()

        new_clean_space = re.sub(r"\s\s+", ' ', predictions[i]).strip()
        new_clean_punc = new_clean_space.translate(str.maketrans("","", string.punctuation)).replace('-', ' ')
        new_clean_punc = re.sub(r"\s\s+", ' ', new_clean_punc).strip()

        if old_clean_punc.lower() == new_clean_punc.lower():
            need_further_proecess_list.append(False)
        else:
            need_further_proecess_list.append(True)
    return need_further_proecess_list


def norm_map(batch, rank):

    torch.cuda.set_device(f"cuda:{(rank or 0) % torch.cuda.device_count()}")

    sentences   = [batch['answer'][i]['text'] for i in range(len(batch['answer']))]
    predictions = model.add_punctuation_capitalization(sentences)

    predictions = [text.replace("<speaker2>", "<Speaker2>").replace("<speaker1>", "<Speaker1>") for text in predictions]

    need_further_proecess_list = check_same(sentences, predictions)

    for i in range(len(sentences)):
        batch['answer'][i]['text']                            = predictions[i]
        batch['other_attributes'][i]['transcription']         = sentences[i]
        batch['other_attributes'][i]['need_further_proecess'] = need_further_proecess_list[i]

    return batch


def main(start=0):
    print("start", flush=True)

    ds = load_from_disk("./datasets/datasets_hf_stage_AudioLLM_v2/datasets_multimodal/train/ASR/gigaspeech_30_ASR")
    N=len(ds)
    print("start", flush=True)
    ds=ds.select(range(start, N))
    
    features                                              = ds.features
    features['other_attributes']['transcription']         = Value('string')
    features['other_attributes']['need_further_proecess'] = Value('bool')

    processed_ds = ds.map(norm_map,
                          batched           = True,
                          with_rank         = True,
                          batch_size        = 1000,
                          features          = features,
                          writer_batch_size = 1,
                          num_proc          = 6,
                          )
    processed_ds.save_to_disk("./workspaces/gigaspeech_30_ASR", num_proc=10)


if __name__ == "__main__":
    set_start_method("spawn")
    main()


