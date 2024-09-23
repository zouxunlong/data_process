import re
import requests
import json
import urllib.request
from bs4 import BeautifulSoup
from multiprocessing import Pool
import os

def yield_results(endpoint, basic_params):
    result = requests.get(endpoint, params=basic_params)
    while True:
        if (result.status_code != 200):
            yield (None, result.json())
            break
        else:
            result_obj = result.json()
            yield (result_obj, None)
            cursor = result_obj.get('cursor', None)
            if cursor is None:
                break
            else:
                params = basic_params.copy()
                params['cursor'] = cursor
                result = requests.get(endpoint, params=params)


def scraping():
    endpoint = "https://archive.org/services/search/v1/scrape"
    basic_params = {"fields": "title",
                    "q": "language:(Malay OR may) AND mediatype:texts"}
    with open("/home/user/data/data_ebook/archive.ms.jsonl", "w" , encoding="utf8") as f_out:
        for batch_index, result in enumerate(yield_results(endpoint, basic_params)):
            for item in result[0]["items"]:
                item["url"] = "https://archive.org/details/{}".format(item["identifier"])
                f_out.write(json.dumps(item, ensure_ascii=False)+"\n")
            print(batch_index, flush=True)


def combine():
    dict={}
    with open("/home/user/data/data_ebook/archive.ms.jsonl") as f_in:
        for line in f_in:
            item=json.loads(line)
            dict[item["identifier"]]=item
    with open("/home/user/data/data_ebook/archive.ms2.jsonl") as f_in2, open("/home/user/data/data_ebook/archive.ms.text.jsonl", "w", encoding="utf8") as f_out:
        for line in f_in2:
            item=json.loads(line)
            item["title"]=dict[item["identifier"]]["title"]
            item["url"]=dict[item["identifier"]]["url"]
            
            f_out.write(json.dumps(item, ensure_ascii=False)+"\n")


def download_text():
    with open("/home/user/data/data_ebook/archive.ms.text.jsonl") as f_in:
        for i, line in enumerate(f_in):

            item=json.loads(line)
            full_text_link=item["full_text_link"]
            title=item["title"]
            print("{}: {}: {} ".format(i, title, full_text_link),flush=True)
            try:
                urllib.request.urlretrieve(full_text_link, "/home/user/data/data_ebook/archive_book.ms/{}.{}".format(title.replace("/", ".")[:100], full_text_link.split(".")[-1]))
            except Exception as e:
                print(e)

def worker(i, full_text_link, title):
    print(os.getpid(), flush=True)
    try:
        urllib.request.urlretrieve(full_text_link, "/home/user/data/data_ebook/archive_book.ms/{}.{}".format(title.replace("/", ".")[:100], full_text_link.split(".")[-1]))
        print("{}: {}: {} ".format(i, title, full_text_link),flush=True)
    except Exception as e:
        print(e, flush=True)

def download_text_multi():

    pool = Pool(20)
    lines= open("/home/user/data/data_ebook/archive.ms.text.jsonl").readlines()
    for i, line in enumerate(lines):
        item=json.loads(line)
        full_text_link=item["full_text_link"]
        title=item["title"]
        pool.apply_async(func=worker, args=(i, full_text_link, title))
    pool.close()
    pool.join()
    print("complete",flush=True)


def unzip(dir):
    return

def extract_txt(file):
    try:
        html=open(file).read()
        if html.startswith("<!DOCTYPE html>"):
            soup = BeautifulSoup(html, features="lxml")
            pre = soup.find('pre')
            if pre:
                text = pre.text
                open(file, "w", encoding="utf8").write(text)
    except Exception as e:
        print("ERROR: {}: {}".format(e, file), flush=True)


def clean(file):
    pattern_punctuation = r"""[ \n\\!?,*.�:;"#$£€%&'()+-/<≤=≠≥>@[\]^_{|}，。、—‘’“”：；【】￥…《》？！（）]"""
    content = open(file).read()
    content = re.sub(
        r'[^a-zA-Z0-9\s\t{}]'.format(
            pattern_punctuation[1:-1],
        ), '', content).strip()
    content = re.sub(r' +', " ", content)
    content="\n".join([line for line in content.split("\n") if line.strip()=="" or (len(line.split())>1 and len(line)/len(re.split(pattern_punctuation, line))>4)])
    content="\n\n".join([" ".join(line.split()) for line in content.split("\n\n") if len(line.split())>8 and len(line)/len(line.split())>3])
    if " dan " in content:
        open(file.replace("/archive_book.ms.txt/", "/archive_book.ms.txt.0/"), "w", encoding="utf8").write(content)


if __name__=="__main__":
    print(os.getpid(), flush=True)
    dir="/home/user/data/data_ebook/archive_book.ms.txt"
    files = os.listdir(dir)
    files.sort()
    for i, file in enumerate(files):
        clean(dir+"/"+file)
        print(i, flush=True)
    print("complete", flush=True)
