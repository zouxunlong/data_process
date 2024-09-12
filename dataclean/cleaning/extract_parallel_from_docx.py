import sys
import pycld2 as cld2
import cld3
import fasttext
import os
import re
from utils_data import Prallel_miner, texts_extract, files_combine, files_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_fasttext = fasttext.load_model('./model/lid.176.bin')

parallel_miner = Prallel_miner(knn_neighbors=6, min_matching_score=0.99, min_cos_sim=0.675,
                               model_path_or_name='./model/labse_bert_model', sort_by_cos=False)

pattern_punctuation = r"""[!?,.:;"#$£€%&'()+-/<≤=≠≥>@[\]^_{|}，。、—‘’“”：；【】￥…《》？！（）]"""
pattern_url = r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
pattern_email = r"[\w\-\.]+@([\w\-]+\.)+[\w\-]{2,4}"
pattern_arabic = r"[\u0600-\u06FF]"
pattern_chinese = r"[\u4e00-\u9fff]"
pattern_tamil = r"[\u0B80-\u0BFF]"
pattern_russian = r"[\u0400-\u04FF]"
pattern_korean = r"[\uac00-\ud7a3]"
pattern_japanese = r"[\u3040-\u30ff\u31f0-\u31ff]"
pattern_vietnamese = r"[àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]"
pattern_emoji = r'[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F64F\U0001F680-\U0001FAFF\U00002702-\U000027B0]'


def lang_detect(text_for_lang_detect1):

    lang_detected = set()

    text_for_lang_detect = ' '.join(re.sub(r"{}|{}|{}".format(
        pattern_url,
        pattern_email,
        pattern_punctuation
    ), " ", text_for_lang_detect1, 0, re.I).split()).strip().lower()

    if text_for_lang_detect:
        if re.search(pattern_arabic, text_for_lang_detect):
            lang_detected.add('ar')
        if re.search(pattern_chinese, text_for_lang_detect):
            lang_detected.add('zh')
        if re.search(pattern_tamil, text_for_lang_detect):
            lang_detected.add('ta')
        if re.search(pattern_russian, text_for_lang_detect):
            lang_detected.add('ru')
        if re.search(pattern_korean, text_for_lang_detect):
            lang_detected.add('ko')
        if re.search(pattern_japanese, text_for_lang_detect):
            lang_detected.add('ja')
        if re.search(pattern_vietnamese, text_for_lang_detect):
            lang_detected.add('vi')

        try:
            lang_by_cld2 = cld2.detect(text_for_lang_detect)[2][0][1]
            lang_by_cld3 = cld3.get_language(text_for_lang_detect)[0]
            lang_by_fasttext = model_fasttext.predict(
                text_for_lang_detect)[0][0][-2:]

            if {"en"} & {lang_by_cld2, lang_by_cld3, lang_by_fasttext}:
                lang_detected.add('en')
            if {'ms'} & {lang_by_cld2, lang_by_cld3, lang_by_fasttext}:
                lang_detected.add('ms')
            if {'id'} & {lang_by_cld2, lang_by_cld3, lang_by_fasttext}:
                lang_detected.add('id')
            if {'th'} & {lang_by_cld2, lang_by_cld3, lang_by_fasttext}:
                lang_detected.add('th')
            if {'vi'} & {lang_by_cld2, lang_by_cld3, lang_by_fasttext}:
                lang_detected.add('vi')
            if {'ta'} & {lang_by_cld2, lang_by_cld3, lang_by_fasttext}:
                lang_detected.add('ta')

        except Exception as err:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno

            print("text_for_lang_detect: ", text_for_lang_detect, flush=True)
            print("Exception type: ", exception_type, flush=True)
            print("File name: ", filename, flush=True)
            print("Line number: ", line_number, flush=True)
            print(err)

    return lang_detected


def allocate_text_by_lang(texts):

    texts_en = []
    texts_ms = []
    texts_zh = []
    texts_ta = []
    texts_vi = []
    texts_th = []
    lang_detected = set()
    for text in texts:
        lang_detecting = lang_detect(text)
        if len(lang_detecting) != 0:
            lang_detected = lang_detecting

        if {"en"} & lang_detected and {"zh"} & lang_detected:
            continue
        elif {"en"} & lang_detected:
            texts_en.append(text)
        elif {"id", "ms"} & lang_detected:
            texts_ms.append(text)
        elif {"zh"} & lang_detected:
            texts_zh.append(text)
        elif {"ta"} & lang_detected:
            texts_ta.append(text)
        elif {"vi"} & lang_detected:
            texts_vi.append(text)
        elif {"th"} & lang_detected:
            texts_th.append(text)

    return {'en': texts_en, 'ms': texts_ms, 'zh': texts_zh, 'ta': texts_ta, 'vi': texts_vi, 'th': texts_th}


def extract_docx(docx_path):

    if not str(docx_path).endswith('.docx'):
        return

    texts = texts_extract(docx_path)

    if not texts:
        return

    text_list_dict = allocate_text_by_lang(texts)

    text_set_dict = parallel_miner.list_to_set(text_list_dict)
    print(text_set_dict['en'][0])

    en_zh_sentence_pair = parallel_miner.sentence_matching(
        text_set_dict['en'], text_set_dict['zh'])
    en_ms_sentence_pair = parallel_miner.sentence_matching(
        text_set_dict['en'], text_set_dict['ms'])
    en_ta_sentence_pair = parallel_miner.sentence_matching(
        text_set_dict['en'], text_set_dict['ta'])

    if en_zh_sentence_pair:
        with open(os.path.splitext(docx_path)[0]+'.en-zh', 'w', encoding='utf8') as fOut:
            for sentence_pair in en_zh_sentence_pair:
                fOut.write("{} ||| {}\n".format(
                    sentence_pair[0], sentence_pair[1]))

    if en_ms_sentence_pair:
        with open(os.path.splitext(docx_path)[0]+'.en-ms', 'w', encoding='utf8') as fOut:
            for sentence_pair in en_ms_sentence_pair:
                fOut.write("{} ||| {}\n".format(
                    sentence_pair[0], sentence_pair[1]))

    if en_ta_sentence_pair:
        with open(os.path.splitext(docx_path)[0]+'.en-ta', 'w', encoding='utf8') as fOut:
            for sentence_pair in en_ta_sentence_pair:
                fOut.write("{} ||| {}\n".format(
                    sentence_pair[0], sentence_pair[1]))

    print(docx_path, flush=True)
    print('en_zh_sentence_pair number:{}'.format(
        len(en_zh_sentence_pair)), flush=True)
    print('en_ms_sentence_pair number:{}'.format(
        len(en_ms_sentence_pair)), flush=True)
    print('en_ta_sentence_pair number:{}'.format(
        len(en_ta_sentence_pair)), flush=True)

    texts.clear()


def extract_dir(root_dir='./file_upload'):

    if not os.path.isdir(root_dir):
        return

    texts = []

    for root, dirs, files in os.walk(root_dir):
        files.sort()

        for i, file in enumerate(files):

            file_path = os.path.join(root, file)

            if file_path.endswith('.docx'):
                texts.extend(texts_extract(file_path))

            if i+1 < len(files) and os.path.splitext(file)[0][:-1].replace(' ', '') == os.path.splitext(files[i+1])[0][:-1].replace(' ', ''):
                continue

            if not texts:
                continue

            text_list_dict = allocate_text_by_lang(texts)

            text_set_dict = parallel_miner.list_to_set(text_list_dict)

            en_zh_sentence_pair = parallel_miner.sentence_matching(
                text_set_dict['en'], text_set_dict['zh'])
            en_ms_sentence_pair = parallel_miner.sentence_matching(
                text_set_dict['en'], text_set_dict['ms'])
            en_ta_sentence_pair = parallel_miner.sentence_matching(
                text_set_dict['en'], text_set_dict['ta'])

            if en_zh_sentence_pair:
                with open(os.path.splitext(file_path)[0]+'.en-zh', 'w', encoding='utf8') as fOut:
                    for sentence_pair in en_zh_sentence_pair:

                        fOut.write("{} ||| {}\n".format(
                            sentence_pair[0], sentence_pair[1]))

            if en_ms_sentence_pair:
                with open(os.path.splitext(file_path)[0]+'.en-ms', 'w', encoding='utf8') as fOut:
                    for sentence_pair in en_ms_sentence_pair:
                        fOut.write("{} ||| {}\n".format(
                            sentence_pair[0], sentence_pair[1]))

            if en_ta_sentence_pair:
                with open(os.path.splitext(file_path)[0]+'.en-ta', 'w', encoding='utf8') as fOut:
                    for sentence_pair in en_ta_sentence_pair:
                        fOut.write("{} ||| {}\n".format(
                            sentence_pair[0], sentence_pair[1]))

            print(file_path, flush=True)
            print('en_zh_sentence_pair number:{}'.format(
                len(en_zh_sentence_pair)), flush=True)
            print('en_ms_sentence_pair number:{}'.format(
                len(en_ms_sentence_pair)), flush=True)
            print('en_ta_sentence_pair number:{}'.format(
                len(en_ta_sentence_pair)), flush=True)

            texts.clear()

if __name__=="__main__":
    # extract_dir('/home/xuanlong/dataclean/data/Batch8(CD8)_extracted')
    # files_combine('/home/xuanlong/dataclean/data/Batch8(CD8)_extracted')
    files_split('/home/xuanlong/dataclean/data')
