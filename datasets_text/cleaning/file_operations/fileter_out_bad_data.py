import os
import re
import string
import time


def fix_Bracket_conjunction(sent):
    if re.search('\)[a-zA-Z]', sent):
        start=re.search('\)[a-zA-Z]', sent).start()
        sent = sent[:start+1]+' '+sent[start+1:]
    return sent


def fix_serial_number(en_sent, non_en_sent):

    if en_sent[0] == non_en_sent[0]:
        if re.match('\d\s', en_sent) and re.match('\d\S', non_en_sent):
            non_en_sent = non_en_sent[:1]+' '+non_en_sent[1:]
        if re.match('\d\s', non_en_sent) and re.match('\d\S', en_sent):
            en_sent = en_sent[:1]+' '+en_sent[1:]
    return en_sent, non_en_sent


def emoji_detected(text_for_emoji_detect):

    if re.search('[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F64F\U0001F680-\U0001FAFF\U00002702-\U000027B0]', text_for_emoji_detect):
        return True

    return False


def non_en_detected(text_for_lang_detect):

    text_for_lang_detect = re.sub(
        "(?i)\w+@\S+\s?|http\S*\s?|www\.\S*\s?|[a-z\.]*\.sg\S*\s?|[0-9]+\s?", "", text_for_lang_detect)
    text_for_lang_detect = text_for_lang_detect.translate(
        str.maketrans('-', ' ', string.punctuation.replace('-', ''))).strip().lower()

    if re.search('[\u4e00-\u9fff\u0B80-\u0BFF\u0400-\u04FF\uac00-\ud7a3\u3040-\u30ff\u31f0-\u31ff\u0A00-\u0A7FàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]', text_for_lang_detect):
        return True

    return False


def filter_line(line,sentence_pair_set):

    sentences = line.strip().split('|')
    if len(sentences) != 2:
        return

    en_sent = ' '.join(re.sub(
        "<Phone Icon>|<Stopwatch>|<br>|<Red>|<Orange>|[\x00-\x1f\x7f-\x9f]|[\u2000-\u200f\u2028-\u202f\u205f-\u206e]|^([a-zA-Z0-9]{1,3}\.|\(?[0-9]+\)|o)\s|^[\^\*\.√§¶†‡+‖#▪●•·-]{1,3}\s?|_+$|\\\\n|</?b>|[\^\*√§¶†‡+‖▪●•·]|\s+(?=[*.,?!，。？！])", "", sentences[0].strip(), flags=re.I).strip().split())
    non_en_sent = ' '.join(re.sub(
        "<Phone Icon>|<Stopwatch>|<br>|<Red>|<Orange>|[\x00-\x1f\x7f-\x9f]|[\u2000-\u200f\u2028-\u202f\u205f-\u206e]|^([a-zA-Z0-9]{1,3}\.|\(?[0-9]+\)|o)\s|^[\^\*\.√§¶†‡+‖#▪●•·-]{1,3}\s?|_+$|\\\\n|</?b>|[\^\*√§¶†‡+‖▪●•·]|\s+(?=[*.,?!，。？！])", "", sentences[-1].strip(), flags=re.I).strip().split())

    if non_en_detected(en_sent):
        return
    if emoji_detected(line):
        return
    if len(re.findall(':', en_sent, re.I)) != len(re.findall(':', non_en_sent, re.I)):
        return
    if len(re.findall('\)', en_sent, re.I)) != len(re.findall('\)', non_en_sent, re.I)):
        return
    if len(re.findall('@', en_sent, re.I)) != len(re.findall('@', non_en_sent, re.I)):
        return
    if len(re.findall('#', en_sent, re.I)) != len(re.findall('#', non_en_sent, re.I)):
        return

    en_sent, non_en_sent = fix_serial_number(en_sent, non_en_sent)
    en_sent = fix_Bracket_conjunction(en_sent)
    non_en_sent = fix_Bracket_conjunction(non_en_sent)

    if len(en_sent.split()) < 3:
        return
    if len(non_en_sent) < 3:
        return

    sentence_pair_set.add((en_sent, non_en_sent))


def filter(file_path):
    with open(file_path) as fIN:
        sentence_pair_set = set()
        for line in fIN:
            filter_line(line, sentence_pair_set)

    with open(file_path, 'w', encoding='utf8') as fOUT:
        sentence_pair_list = list(sentence_pair_set)
        sentence_pair_list_sorted = sorted(sentence_pair_list)
        for en_sent, non_en_sent in sentence_pair_list_sorted:
            fOUT.write("{} | {}\n".format(en_sent, non_en_sent))

    print("finished " + file_path)


def main(rootdir):
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if os.path.splitext(file)[1] in {'.en-ta', '.en-zh', '.en-vi', '.en-ms', '.en-id'}:
                file_path = os.path.join(root, file)
                filter(file_path)


if __name__ == '__main__':
    start_time = time.time()
    rootdir = '/home/xuanlong/dataclean/data/MCI_combined'
    main(rootdir)
    print("--- %s seconds ---" % (time.time() - start_time), flush=True)
