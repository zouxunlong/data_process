import random
import re


class Utils:
    pattern_punctuation = r"""([!?,.:;"#$£€%&'()+-/<≤=≠≥>@[\]^_{|}，。、—‘’“”：；【】￥…《》？！（）\u0964\u0965\uAAF1\uAAF0\uABEB\uABEC\uABED\uABEE\uABEF\u1C7E\u1C7F])"""
    pattern_number = r'([0-9]+([,.:/-][0-9]+)*)'
    pattern_url = r"(((http|https|ftp)\://)?([a-zA-Z0-9\.\-]+(\:[a-zA-Z0-9\.&amp;%\$\-]+)*@)*((25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9])\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9]|0)\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9]|0)\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[0-9])|localhost|([a-zA-Z0-9\-]+\.)*[a-zA-Z0-9\-]+\.(com|edu|gov|int|mil|net|org|biz|arpa|info|name|pro|aero|coop|museum|[a-zA-Z]{2}))(\:[0-9]+)*(/($|[a-zA-Z0-9\.\,\?\'\\\+&amp;%\$#\=~_\-]+))*$)"
    pattern_email = r"([a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+){0,4}@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+){0,4}$)"
    pattern_arabic = r"[\u0600-\u06FF]"
    pattern_chinese = r"[\u4e00-\u9fff]"
    pattern_tamil = r"[\u0B80-\u0BFF]"
    pattern_thai = r"[\u0E00-\u0E7F]"
    pattern_russian = r"[\u0400-\u04FF]"
    pattern_korean = r"[\uac00-\ud7a3]"
    pattern_japanese = r"[\u3040-\u30ff\u31f0-\u31ff]"
    pattern_vietnamese = r"[àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]"
    pattern_emoji = r'[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F64F\U0001F680-\U0001FAFF\U00002702-\U000027B0]'
    pattern_initials=r"((?<=([^\u0B80-\u0BFF]))(([\u0B80-\u0BFF]{1,3}\.)+([\u0B80-\u0BFF]{1,3})?|திரு\.|திருமதி\.|செல்வி\.|டாக்டர்\.)|^(([\u0B80-\u0BFF]{1,3}\.)+([\u0B80-\u0BFF]{1,3})?|திரு\.|திருமதி\.|செல்வி\.|டாக்டர்\.))"

    punc2sent_len = {'.':5, '?':5, '!':5, '。':5, '？':5, '！':5, ';':10, '；':10, ':':20, '：':20, ',':50, '，':50}
    end_set=set(['”',')',']','}','』','」','》','）','】','｝','〕','﹚','.','?','!', '。','？','！','’',';','；',':','：'])
    # punc = ['.', '?', '!', ';', ':', ',', '(', ')', '"', '\'', '...', '+', '*', '&', '%', '$', '@', '#', '£', '€', '_', '<', '≤', '≠', '≥', '>', '[', ']', '-', '~', '=', '^', '{', '}', '‘', '’', '\\', '/', '“']
    punc = '.?!;:,()"\'...+*&%$@#£€_<≤≠≥>[]-~=^{}‘’\\/“'


def check_patterns(segment):
    if bool(re.search(Utils.pattern_url, segment)):
        segment=re.sub(Utils.pattern_url, r' \1 ', segment)
    elif bool(re.search(Utils.pattern_email, segment)):
        segment=re.sub(Utils.pattern_email, r' \1 ', segment)
    elif bool(re.search(Utils.pattern_number, segment)):
        segment=re.sub(Utils.pattern_number, r' \1 ', segment)
    elif bool(re.search(Utils.pattern_initials, segment)):
        segment=re.sub(Utils.pattern_initials, r' \1 ', segment)
    return segment


def tokenize(input_text):
    def _get_tokens(input_text):
        for segment in input_text.split():
            result=check_patterns(segment)
            if result.count(' ')==2:
                ws=result.split()
                if len(ws)==1:
                    yield result.strip()
                else:
                    for w in ws:
                        yield from _get_tokens(w)
            elif result.count(' ')==0:
                word_chars = ''
                for char in result:
                    if char in Utils.punc:
                        if word_chars:
                            yield word_chars
                            word_chars = ''
                        yield char
                    else:
                        word_chars += char
                if word_chars:
                    yield word_chars
    return list(_get_tokens(input_text))


def split_utt(its0):
    mask=[False]*len(its0)
    its=its0
    i=0
    slen=0
    inXML = False
    while i<len(its):
        w=its[i]
        slen+=1
        # check XML-markup
        if w=='<n' or w=='<np':
            inXML = True
        elif w.endswith('</n>') or w.endswith('</np>'):
            inXML = False
        # too long, force split
        if slen>1024 and not inXML:
            mask[i]=True
            i+=1
            slen=0
            continue
        # at punctuation
        if w in Utils.punc2sent_len and not inXML:
            while i+1<len(its) and (its[i+1] in Utils.end_set or (its[i+1]=='"' and its[0:i+2].count('"')%2==0)):
                i+=1
                slen+=1
            if slen>=Utils.punc2sent_len[w]:
                mask[i]=True
                i+=1
                slen=0
                continue
        i+=1
    # never split at EOL
    mask[-1]=False

    S=' '.join([(word+'\n' if mask[i] else word) for i, word in enumerate(its0)])
    return [sent.split() for sent in S.split('\n')]


def split_sents(L, ):
    L=tokenize(L)
    outs = split_utt(L)
    lines=[' '.join(its)+' ' for its in outs]
    print(len(lines), flush=True)
    with open("/home/xunlong/xunlong_working_repo/_data_in_processing/mt_data/ta.txt", 'a', encoding='utf8') as file_out:
        for line in lines:
            file_out.write(line+'\n')


def main(file):
    lines=open(file).readlines()
    random.shuffle(lines)
    split_sents(" ".join(lines))


if __name__=="__main__":
    from datasets import load_from_disk
    ds=load_from_disk("/home/xunlong/xunlong_working_repo/_data_in_processing/mt_data/newslink_ta.hf")
    for sample in ds:
        split_sents(" ".join(sample['text'].split('\n')))

