import os
from sentsplit.segment import SentSplit
from bi_text_miner import Bi_text_miner

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
torch.cuda.set_device(1)

bi_text_miner = Bi_text_miner(knn_neighbors=4, min_matching_score=0.99, min_cos_sim=0.6,
                              model_path_or_name='../model/labse_bert_model', sort_by_cos=False)


sent_splitter_en = SentSplit('en', maxcut=80, strip_spaces=True)
sent_splitter_zh = SentSplit('zh', maxcut=28, strip_spaces=True)


def extract_en_sentences(paragraph):
    sentences = sent_splitter_en.segment(paragraph)
    return sentences

def extract_zh_sentences(paragraph):
    sentences = sent_splitter_zh.segment(paragraph)
    return sentences

def main(file_in_1='/home/xuanlong/dataclean/data/three_body.en',file_in_2='/home/xuanlong/dataclean/data/three_body.zh'):

    os.chdir(os.path.dirname(__file__))

    with open(file_in_1, 'r', encoding='UTF-8-sig') as f_in_en, \
            open(file_in_2, 'r', encoding='UTF-8-sig') as f_in_zh, \
            open(os.path.splitext(file_in_1)[0]+'.aligned'+os.path.splitext(file_in_1)[1], 'a', encoding='utf8') as f_out_en, \
            open(os.path.splitext(file_in_2)[0]+'.aligned'+os.path.splitext(file_in_2)[1], 'a', encoding='utf8') as f_out_zh:

        for (i, sentence_src), (j, sentence_tgt) in zip(enumerate(f_in_en), enumerate(f_in_zh)):
            sentence_src = sentence_src.strip()
            sentence_tgt = sentence_tgt.strip()
            sentences_src = extract_en_sentences(sentence_src)
            sentences_tgt = extract_zh_sentences(sentence_tgt)

            text_list_dict = {'en': sentences_src, 'id': sentences_tgt}
            try:
                text_set_dict = bi_text_miner.list_to_set(text_list_dict)
            except:
                print('Error Occurs while list to set')
                continue
            en_id_sentence_pair = bi_text_miner.sentence_matching(
                text_set_dict['en'], text_set_dict['id'])

            for sentence_pair in en_id_sentence_pair:
                f_out_en.write(sentence_pair[0]+'\n')
                f_out_zh.write(sentence_pair[1]+'\n')
            


if __name__ == "__main__":
    main('/home/xuanlong/dataclean/data/three_body.en','/home/xuanlong/dataclean/data/three_body.zh')
