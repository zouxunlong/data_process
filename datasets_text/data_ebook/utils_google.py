import json
import re
import urllib.request
import fitz
import os


def download_text():
    with open("/home/user/data/data_ebook/text_book.id.jsonl") as f_in:
        for i, line in enumerate(f_in):
            item = json.loads(line)
            book_link = item["book_link"]
            book_id = book_link.split("/")[-2]
            title = item["title"]
            print("{}: {}: {} ".format(i, title, book_link), flush=True)
            try:
                urllib.request.urlretrieve("https://drive.usercontent.google.com/download?id={}&export=download&authuser=0".format(
                    book_id), "/home/user/data/data_ebook/google_book.id/{}.pdf".format(title))
            except Exception as e:
                print(e)


def extract_txt():

    dir = "/home/zxl/ssd/pdf2text/google_book.id/"
    files = os.listdir(dir)
    for file in files:
        if os.path.exists(dir+file.replace(".pdf", ".2.txt")):
            continue
        try:
            doc = fitz.open(dir+file)
            text = ""
            for i, page in enumerate(doc):
                text += page.get_text() + "\n\n"
            open(dir+file.replace(".pdf", ".2.txt"),
                 "w", encoding="utf8").write(text)
            print("complete {}".format(file), flush=True)
        except Exception as e:
            print(e, flush=True)
            print("error on file: {}".format(file), flush=True)


def clean(file):
    pattern_punctuation = r"""[ \n\\!?,*.�:;"#$£€%&'()+-/<≤=≠≥>@[\]^_{|}，。、—‘’“”：；【】￥…《》？！（）]"""
    content = open(file).read()
    content = re.sub(
        r'[^a-zA-Z0-9\s\t{}{}]'.format(
            pattern_punctuation[1:-1],
        ), '', content).strip()
    content = re.sub(r' +', " ", content)
    content="\n".join([line for line in content.split("\n") if line.strip()=="" or (len(line.split())>1 and len(line)/len(re.split(pattern_punctuation, line))>3)])
    content="\n\n".join([" ".join(line.split()) for line in content.split("\n\n") if len(line.split())>8 and len(line)/len(line.split())>3])
    if " dan " in content:
        open(file.replace("/google_book.id.0.txt/", "/google_book.id.txt/"), "w", encoding="utf8").write(content)



def clean_main():
    dir = "/home/user/data/data_ebook/google_book.id.1.txt/"
    files = os.listdir(dir)
    files.sort()
    for i, file in enumerate(files):
        clean(dir+file)
        print(i, flush=True)
    print("complete", flush=True)


def count_tokens(dir):
    count=0
    files = os.listdir(dir)
    files.sort()
    for file in files:
        content = open(dir+"/"+file).read()
        count+=len(content.split())
        print(count, flush=True)
    print("complete", flush=True)

def count_tokens_file(file):
    count=0
    with open(file) as f:
        for i, line in enumerate(f):
            if i%10000==0:
                print(i, flush=True)
            item=json.loads(line)
            content = item["text"]
            count+=len(content.split())
    print(count, flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    dir = "/home/user/data/data_ebook/archive_book.ms.txt.0"
    count_tokens(dir)


