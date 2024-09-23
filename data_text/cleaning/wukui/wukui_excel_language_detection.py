import pandas as pd
import sys
import pycld2 as cld2
import cld3
import fasttext
import re


model_fasttext = fasttext.load_model('./model/lid.176.bin')


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
            lang_by_cld2 = cld2.detect(text_for_lang_detect)[2][0][1][:2]
            lang_by_cld3 = cld3.get_language(text_for_lang_detect)[0][:2]
            
            # lang_by_fasttext = model_fasttext.predict(
            #     text_for_lang_detect)[0][0][-2:]
            # lang_detected.add(lang_by_cld2)
            # lang_detected.add(lang_by_cld3)
            # lang_detected.add(lang_by_fasttext)

            if {"en"} & {lang_by_cld2, lang_by_cld3}:
                lang_detected.add('en')
            if {'ms','id'} & {lang_by_cld2, lang_by_cld3}:
                lang_detected.add('ms')
            # if {'id'} & {lang_by_cld2, lang_by_cld3}:
            #     lang_detected.add('id')
            if {'th'} & {lang_by_cld2, lang_by_cld3}:
                lang_detected.add('th')
            if {'vi'} & {lang_by_cld2, lang_by_cld3}:
                lang_detected.add('vi')
            if {'ta'} & {lang_by_cld2, lang_by_cld3}:
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

df = pd.read_excel('Sample Data.xlsx', header=None)

dfList = []

for i, [sentence0, sentence1, sentence2, sentence3, sentence4] in enumerate(df.loc[0:].values):

    lang_set = lang_detect(sentence1)



    dfList.append([sentence0, sentence1, sentence2, sentence3, lang_set])

df_new =  pd.DataFrame(dfList)
df_new.to_excel('output.xlsx', index=False, header=False)
