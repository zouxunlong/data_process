
from glob import glob
import os
import json
from itertools import groupby
import re
from docx import Document
from docx.oxml.shared import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run


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


def get_all_paragraphs(node):
    def _get(node):
        for child in node:
            if child.tag == qn('w:p'):
                yield Paragraph(child, node)
            yield from _get(child)
    return list(_get(node._element))


def docx2paragraphs(docx_path):

    if not str(docx_path).endswith('.docx'):
        return []

    wordDoc = Document(docx_path)
    items = get_all_paragraphs(wordDoc)

    texts = [item.text.strip()
             for item in items if item.text.strip()]

    return texts


if __name__ == "__main__":

    speakers=set()
    files=glob("/data/xunlong/NLB_data_preparation/data_NLB/test/docx/*")

    for file in files:

        texts = docx2paragraphs(file)
        for sentence in texts:
            sentence=re.sub(r'\t+', '\t', sentence).strip().replace(u'\xa0', u' ')
            if len(sentence.split("\t"))>1:
                speaker = sentence.split("\t")[0].strip(" .+:0/`").upper()
                speakers.add(speaker.upper())
    speakers={speaker:speaker for speaker in speakers}
    open("speakers.json", "w").write(json.dumps(speakers, indent=4))

