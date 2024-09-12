import os
from pathlib import Path
import socket
import plac
import re
from docx import Document
from docx.oxml.shared import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run, _Text


pattern_punctuation = r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~，。、‘’“”：；【】·！￥★…《》？！（）—]"""
pattern_url = r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
pattern_email = r"[\w\-\.]+@([\w\-]+\.)+[\w\-]{2,4}"
pattern_arabic = r"[\u0600-\u06FF]"
pattern_chinese = r"[\u4e00-\u9fff]"
pattern_tamil = r"[\u0B80-\u0BFF]"
pattern_russian = r"[\u0400-\u04FF]"
pattern_korean = r"[\uac00-\ud7a3]"
pattern_japanese = r"[\u3040-\u30ff\u31f0-\u31ff]"
pattern_vietnamese = r"[àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]"


def non_en_detect(text_for_lang_detect):

    lang_detected = set()

    text_for_lang_detect = ' '.join(re.sub("{}|{}|{}".format(
        pattern_url, pattern_email, pattern_punctuation), " ", text_for_lang_detect, 0, re.I).split()).strip().lower()

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

    return lang_detected


def wh_2022_6_14_api(src_sent, s):
    s.sendall(bytes(src_sent+"\n", encoding='utf8'))
    tgt_sent = s.recv(1024).decode('UTF-8')
    return tgt_sent


def translate(src, s):
    tgt = ''.join([wh_2022_6_14_api(line, s)
                  for line in src.split('\n')]).strip()
    return tgt


def get_all_texts(node):
    def _get(node):
        for child in node:
            if child.tag == qn('w:t'):
                yield _Text(child)._t
            yield from _get(child)
    return list(_get(node._element))


def get_all_runs(node):
    def _get(node):
        for child in node:
            if child.tag == qn('w:r'):
                yield Run(child, node)
            yield from _get(child)
    return list(_get(node._element))


def get_all_paragraphs(node):
    def _get(node):
        for child in node:
            if child.tag == qn('w:p'):
                yield Paragraph(child, node)
            yield from _get(child)
    return list(_get(node._element))


def get_paragraph_runs(paragraph):
    def _get(node):
        for child in node:
            if not (child.tag == qn('w:drawing') or child.tag == qn('w:pict')):
                if child.tag == qn('w:r'):
                    yield Run(child, node)
                yield from _get(child)
    return list(_get(paragraph._element))


def get_paragraph_text(paragraph):
    text = ''
    for run in paragraph.runs:
        text += run.text
    return text


def set_paragraph_text(paragraph, text):
    runs = paragraph.runs
    for run in runs:
        if run.text.strip():
            run._r.getparent().remove(run._r)
    for child in paragraph._element:
        if child.tag == qn('w:hyperlink'):
            if len(child) == 0:
                child.getparent().remove(child)
    paragraph.add_run(text)


Paragraph.runs = property(fget=lambda self: get_paragraph_runs(self))
Paragraph.text = property(fget=lambda self: get_paragraph_text(self),
                          fset=lambda self, text: set_paragraph_text(self, text))


def do_translation(file_path, src2tgt):

    try:
        wordDoc = Document(file_path)
        items = get_all_paragraphs(wordDoc)
        for section in wordDoc.sections:
            header = section.header
            footer = section.footer
            items += get_all_paragraphs(header)
            items += get_all_paragraphs(footer)

        texts = [item.text for item in items if item.text.strip()]

        print(texts, flush=True)

        if src2tgt == 'en2zh':
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_en2zh:
                socket_en2zh.connect(('10.2.56.190', 10228))
                for item in items:
                    if not non_en_detect(item.text.strip()):
                        item.text = translate(item.text, socket_en2zh)

        elif src2tgt == 'zh2en':
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_zh2en:
                socket_zh2en.connect(('10.2.56.190', 10177))
                for item in items:
                    if 'zh' in non_en_detect(item.text.strip()):
                        item.text = translate(item.text, socket_zh2en)

        wordDoc.save(file_path)
        print('docx translation finished')
        return 0

    except Exception as err:
        return err


@ plac.pos('file_path', "file path", type=str)
def main(file_path='./demo.docx'):
    do_translation(file_path, 'en2zh')


if __name__ == "__main__":
    plac.call(main)
