#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, April 25th 2024, 2:11:39 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
#
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
# From Xunlong's code
###


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

    source_path = "/mnt/home/zoux/datasets/NLB/test/docx"
    dst_path = "/mnt/home/zoux/datasets/NLB/test/jsonl"

    speakers = json.load(open("speakers.test.json"))

    filesnames = os.listdir(source_path)
    filesnames.sort()

    for key, value in groupby(filesnames, key=lambda x: x.split("-")[0]):
        files = list(value)
        files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))

        for filename in files:

            isstarted = False
            entrys = []
            texts = docx2paragraphs("{}/{}".format(source_path, filename))

            for sentence in texts:

                sentence = re.sub(r'\t+', '\t', sentence).strip().replace(u'\xa0', u' ')
                if not isstarted:
                    if int(filename.split("-")[1].split(".")[0]) == 1 \
                            or filename in ["001891-07.docx", "003338-02.docx", "000301-18.docx"] \
                            or filename.split("-")[0] in ["000299", "001951"]:

                        if len(sentence.split("\t")) > 1:
                            speaker = sentence.split("\t")[0].strip(" .+:0/`").upper()
                            if speaker in speakers:
                                current_speaker = speakers[speaker]
                                isstarted = True
                    else:
                        isstarted = True
                        speaker = sentence.split("\t")[0].strip(" .+:0/`").upper()
                        if speaker in speakers:
                            current_speaker = speakers[speaker]
                        else:
                            print("Unknown speaker: {} ".format(speaker), flush=True)
                            print("{}: {}: {}".format(filename, current_speaker, sentence), flush=True)

                if isstarted:
                    speaker = sentence.split("\t")[0].strip(" .+:0/`").upper()
                    if speaker in speakers:
                        current_speaker = speakers[speaker]
                        utterance = " ".join(" ".join(sentence.split("\t")[1:]).split()).strip()
                    else:
                        utterance = " ".join(sentence.split()).strip()

                    utterance = re.sub('\s?\[[^\[\]]*\]\s?', " ", utterance).translate(str.maketrans('‘’“”', '\'\'""'))
                    if not "End of Reel" in utterance:
                        entrys.append({'speaker': current_speaker.strip(), 'utterance': utterance})

            with open('{}/{}'.format(dst_path, filename.replace(".docx", ".jsonl")), 'w') as outfile:
                for entry in entrys:
                    outfile.write(json.dumps(entry, ensure_ascii=False)+'\n')

    print("Done", flush=True)
