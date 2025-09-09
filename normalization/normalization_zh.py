from transformers import AutoModelForTokenClassification,AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk
import re
import os
from glob import glob
import warnings 
import fire
import torch
from torch.utils.data import Dataset
from zhpr.core import get_tokenizer,make_model_config
from multiprocess import set_start_method

class DocumentDataset(Dataset):
    def __init__(self, documents:list[str]) -> None:
        super().__init__()
        self.tokenizer = get_tokenizer()
        self.config = make_model_config()
        self.documents = documents
        self.data = [{'tokens': list(document)} for document in self.documents]
        if self.data:
            self.window_size = max([len(d['tokens']) for d in self.data])+1

    def __getitem__(self, index):
        data = self.data[index]
        tokens = self.tokenizer.convert_tokens_to_ids(data['tokens'])
        while len(tokens) < self.window_size:
            tokens.append(self.tokenizer.pad_token_id)
        return torch.tensor(tokens)

    def __len__(self):
        return len(self.data)

def decode_pred(token_ners):
    out = []
    for token_ner in token_ners:
        out.append(token_ner[0])
        if token_ner[-1] != 'O':
            out.append(token_ner[-1][-1])
    return out

warnings.filterwarnings('ignore')
model_name = 'p208p2002/zh-wiki-punctuation-restore'
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pattern_punctuation = r"""[!?,*.:;"#$£€%&'()+-/<≤=≠≥>@[\]^_{|}，。、—‘’“”：；【】￥…《》？！（）]"""


def check_same(sentences, predictions):
    need_further_process_list = []
    for i in range(len(sentences)):
        old_clean_space = re.sub(r"\s", '', sentences[i]).strip()
        old_clean_punc = old_clean_space.translate(str.maketrans("","", pattern_punctuation)).replace('\n', '')
        old_clean_punc = re.sub(r"\s", '', old_clean_punc).strip()

        new_clean_space = re.sub(r"\s", '', predictions[i]).strip()
        new_clean_punc = new_clean_space.translate(str.maketrans("","", pattern_punctuation)).replace('\n', '')
        new_clean_punc = re.sub(r"\s", '', new_clean_punc).strip()

        if old_clean_punc.lower() == new_clean_punc.lower():
            need_further_process_list.append(False)
        else:
            need_further_process_list.append(True)
    return need_further_process_list


def predict_step(batch_input_ids, model,tokenizer):
    batch_predictions = []

    encodings = {'input_ids': batch_input_ids}
    output = model(**encodings)
    predicted_token_class_id_batch = output['logits'].argmax(-1)
    for predicted_token_class_ids, input_ids in zip(predicted_token_class_id_batch, batch_input_ids):
        out=[]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        input_ids = input_ids.tolist()
        try:
            input_id_pad_start = input_ids.index(tokenizer.pad_token_id)
        except:
            input_id_pad_start = len(input_ids)
        input_ids = input_ids[:input_id_pad_start]
        tokens = tokens[:input_id_pad_start]

        # predicted_token_class_ids
        predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids]
        predicted_tokens_classes = predicted_tokens_classes[:input_id_pad_start]

        for token,ner in zip(tokens,predicted_tokens_classes):
            out.append(token)
            if ner != 'O':
                out.append(ner[-1])

        prediction = ''.join(out).replace('<[UNK]peaker2>:','<Speaker2>: ').replace('<[UNK]peaker1>:','<Speaker1>: ').replace('[UNK]', '').replace('<。peaker1>:','<Speaker1>: ')
        batch_predictions.append(prediction)
    return batch_predictions


def norm_map(batch, rank):

    torch.cuda.set_device(f"cuda:{(rank or 0) % torch.cuda.device_count()}")
    model.to("cuda")

    original_answers = [''.join(answer['text'].split()) for answer in batch['answer']]
    need_process_answers = [(index, answer) for index, answer in enumerate(original_answers) if re.search(pattern_punctuation, answer.replace("<Speaker1>:", "").replace("<Speaker2>:", "")) is None]
    indexs, sentences = zip(*need_process_answers) if len(need_process_answers) > 0 else ([], [])

    dataset = DocumentDataset(sentences)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1000)

    predictions = []
    for batch_input_ids in dataloader:
        batch_input_ids   = batch_input_ids.to("cuda")
        batch_predictions = predict_step(batch_input_ids, model, tokenizer)
        predictions.extend(batch_predictions)

    need_further_process_list = check_same(sentences, predictions)


    for i, index in enumerate(indexs):
        if need_further_process_list[i]:
            batch['answer'][index]['text'] = sentences[i]
            batch['other_attributes'][index]['need_further_process'] = need_further_process_list[i]
        else:
            batch['answer'][index]['text'] = predictions[i]
            batch['other_attributes'][index]['need_further_process'] = need_further_process_list[i]

    return batch


def main(rev=False):
    input_paths=glob("/data/projects/13003558/zoux/datasets/datasets_hf_stage_AudioLLM_v2_normalized/datasets_multimodal/train/ASR/*_zh_30*_ASR")

    for input_path in sorted(input_paths, reverse=rev):
        output_path = input_path.replace("/ASR/","/ASR_normalized/")

        if os.path.exists(output_path):
            print(f"{output_path} exists, skip", flush=True)
            continue


        print(f"start {input_path}", flush=True)

        ds = load_from_disk(input_path)

        normalized_ds = ds.map(norm_map,
                            batched           = True,
                            with_rank         = True,
                            batch_size        = 1000,
                            writer_batch_size = 1000,
                            num_proc          = 8,
                            )
        normalized_ds.save_to_disk(output_path, num_proc=4)

if __name__ == "__main__":
    set_start_method("spawn")
    fire.Fire(main)