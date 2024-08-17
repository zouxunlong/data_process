import requests
import re
import json
import scrapy
from urllib import request
import os
import fitz

def get_dates(output_path='sittingDate.txt'):

    sittingDates = set()
    base_url = "https://sprs.parl.gov.sg/search/searchResult"
    body = {
        "keyword": "",
        "fromday": "05",
        "frommonth": "04",
        "fromyear": "2024",
        "today": "05",
        "tomonth": "04",
        "toyear": "2024",
        "dateRange": "* TO NOW",
        "reportContent": "with all the words",
        "parliamentNo": "",
        "selectedSort": "date_dt desc",
        "portfolio": [], "mpName": "",
        "rsSelected": "",
        "lang": "cmn,msa,tam",
        "startIndex": "40",
        "endIndex": "59",
        "titleChecked": "false",
        "footNoteChecked": "false",
        "ministrySelected": []
    }
    for start in range(0, 3100, 20):
        body["startIndex"] = start
        body["endIndex"] = start+20
        results = requests.post(base_url, json=body)
        items = json.loads(results.text)
        for item in items:
            # print(item["sittingDate"], flush=True)
            sittingDates.add(item["sittingDate"])

    for date in sorted(sittingDates):
        open(output_path, "w").write(date+"\n")


def generate_pairs1(inputfile):
    lines = open(inputfile).readlines()
    for line in lines:
        item = json.loads(line)
        print(item["sittingDate"], flush=True)

        text_node = scrapy.Selector(text=item["htmlFullContent"])

        raw_text = ''.join(text_node.xpath("//div[./p or ./span]//text()").getall()).replace('\r\n', " ").replace('\t', " ").replace('\xa0', " ").replace('    ', "\n")
        raw_text = re.sub(r"Column: [0-9]{1,4}", " ", raw_text) 
        raw_text = re.sub(r"Column No : [0-9]{1,4}", " ", raw_text) 
        raw_text = re.sub(r"[0-9]{1,2}\.[0-9]{2} (pm|am)", " ", raw_text) 
        raw_text = re.sub(r" +", " ", raw_text) 
        raw_text = re.sub(r"\n*\*Cols\. [0-9-; \.]+\n*", r"\n", raw_text)
        lines = re.sub(r"\n*Page: [0-9]*\n*", r"\n", raw_text).splitlines() 

        texts = [line.strip() for line in lines if line.strip()]

        texts = ['\n'+text
                 if re.search(r"^[a-zA-Z0-9 ()\.\[\]-]*:", text) 
                 or re.search(r"\( ?in (mandarin|malay|tamil|english)", text, flags=re.I)
                 else
                 text
                 for text in texts]

        text = "\n".join(texts)
        open("text/{}.txt".format(item["sittingDate"]), "w").write(text)

        snippets = [snippet.strip() for snippet in text.split("\n\n") if
                    re.search(r"\(in (mandarin|malay|tamil)", snippet, flags=re.I) and re.search(r"vernacular speech", snippet, flags=re.I)]
        open("text/{}.snippet".format(item["sittingDate"]), "w").write("\n\n".join(snippets))

        a_nodes = scrapy.Selector(text=item["htmlFullContent"]).xpath(
            "//a[contains(@href,'.pdf') and (contains(.//text(),'Appendix A') or contains(.//text(),'Vernacular Speech'))]")
        if a_nodes:
            hrefs = [a_node.xpath("./@href").get() for a_node in a_nodes]
            href = "\n".join(["https://sprs.parl.gov.sg{}".format(href) for href in hrefs])+"\n"
            open("text/{}.href".format(item["sittingDate"]), "w").write(href)
        try:
            assert len(snippets) == len(hrefs)
        except:
            print("ERROR: {}".format(item["sittingDate"]), flush=True)


def generate_pairs2(inputfile):
    lines = open(inputfile).readlines()
    for line in lines:
        item = json.loads(line)
        print(item["sittingDate"])
        texts=[]
        hrefs=[]
        for section in item["takesSectionVOList"]:
            text_nodes = scrapy.Selector(text=section["content"]).xpath("//p")
            if text_nodes:
                lines = ["\n"+''.join(text_node.xpath(".//text()").getall()).replace('\t', " ").replace('\xa0', " ").strip() if text_node.xpath('.//strong') and text_node.xpath('./text()')
                         else ''.join(text_node.xpath(".//text()").getall()).replace('\t', " ").replace('\xa0', " ").strip() for text_node in text_nodes]
                lines=[line for line in lines if line.strip()]
                texts.extend(["\n"+text if re.search(r"\(in (mandarin|malay|tamil|english)", text, flags=re.I) and not text.startswith("\n") else text for text in lines])

            a_nodes = scrapy.Selector(text=section["content"]).xpath(
                "//a[contains(@href,'.pdf') and contains(text(),'Vernacular Speech')]")
            if a_nodes:
                hrefs.extend([a_node.xpath("./@href").get() for a_node in a_nodes])
        
        href = "\n".join(["https://sprs.parl.gov.sg{}".format(href) for href in hrefs])+"\n"
        open("text/{}.href".format(item["sittingDate"]), "w").write(href)

        text = "\n".join(texts)
        open("text/{}.txt".format(item["sittingDate"]), "w").write(text)
        
        snippets = [snippet.strip() for snippet in text.split("\n\n") if 
                    re.search(r"\(in (mandarin|malay|tamil)", snippet, flags=re.I) and re.search(r"vernacular speech", snippet, flags=re.I)]
        open("text/{}.snippet".format(item["sittingDate"]), "w").write("\n\n".join(snippets))

        try:
            assert len(snippets) == len(hrefs)
        except:
            print("ERROR: {}".format(item["sittingDate"]), flush=True)


def download_pdf(dir):
    files = os.listdir(dir)
    for file in files:
        if file.endswith(".href"):
            file_path = dir+"/"+file
            urls = open(file_path).readlines()
            for i, url in enumerate(urls):
                output_dir = file_path[:-5]
                os.makedirs(output_dir, exist_ok=True)
                if os.path.exists("{}/{}.pdf".format(file_path[:-5], i)):
                    continue
                try:
                    print("start {}/{}.pdf".format(file_path[:-5], i))
                    request.urlretrieve(
                        url[:-1].replace(' ', '%20'), "{}/{}.pdf".format(file_path[:-5], i))
                except Exception as e:
                    print("----------", flush=True)
                    print(e, flush=True)
                    print("{}/{}.pdf".format(file_path[:-5], i), flush=True)
                    print(url, flush=True)


def check_pdf2snippet():
    for parent_dir, dirs, files in os.walk("text"):
        for dir in dirs:
            count1=len(open("text/{}.snippet".format(dir)).read().split("\n\n"))
            count2=len(os.listdir("text/{}".format(dir)))
            try:
                assert 2*count1==count2
            except:
                print("pdfs 2 snippets not match:{}".format(dir), flush=True)


def check_href2snippet():
    for parent_dir, dirs, files in os.walk("text"):
        for dir in dirs:
            count1=len(open("text/{}.snippet".format(dir)).read().split("\n\n"))
            count2=len(open("text/{}.href".format(dir)).readlines())
            try:
                assert count1==count2
            except:
                print("hrefs 2 snippets not match:{}".format(dir), flush=True)


def pdf2text():
    dir = "text"
    for parent_dir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".pdf"):
                output_path = parent_dir+"/"+file.replace(".pdf", ".txt")
                # if os.path.exists(output_path):
                #     continue
                try:
                    doc = fitz.open(parent_dir+"/"+file)
                    text = ""
                    for i, page in enumerate(doc):
                        text += page.get_text()
                    open(output_path, "w", encoding="utf8").write(text)
                    print("complete {}".format(output_path), flush=True)
                except Exception as e:
                    print(e, flush=True)
                    print("error on {}".format(output_path), flush=True)


def reformat1():
    dir = "text"
    for parent_dir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".txt"):
                output_path = parent_dir+"/"+file
                content = open(output_path).read()
                content = re.sub(r"\n(\S)", r"\1", content)
                content = re.sub(r" +", " ", content)
                open(output_path, "w", encoding="utf8").write(content)


def reformat2():
    dir = "text"
    for parent_dir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".txt"):
                output_path = parent_dir+"/"+file
                lines = open(output_path).readlines()
                content = "\n".join([line.strip() for line in lines if line.strip()])+"\n"
                open(output_path, "w", encoding="utf8").write(content)


def reformat3():
    dir = "text"
    for parent_dir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".txt"):
                output_path = parent_dir+"/"+file
                lines = open(output_path).readlines()
                content = "\n".join([line.strip() for line in lines if not line in ["Appendix A\n", "1\n", "2\n"]])+"\n"
                open(output_path, "w", encoding="utf8").write(content)


def match():
    snippet_files=[file for file in os.listdir("text") if file.endswith(".snippet")]
    print(len(snippet_files), flush=True)
    for file in snippet_files:
        print(file, flush=True)
        snippets=open("text/{}".format(file)).read().split("\n\n")
        for i, snippet in enumerate(snippets):
            matching_path="text/{}/{}.txt".format(file.split(".")[0], i)
            if os.path.exists(matching_path):
                if re.search(r"\( ?in mandarin", snippet, flags=re.I):
                    target_lang="zh"
                elif re.search(r"\( ?in malay", snippet, flags=re.I):
                    target_lang="ms"
                elif re.search(r"\( ?in tamil", snippet, flags=re.I):
                    target_lang="ta"
                text_en=re.split(r"\[[^\]]+\]", snippet, 1)[-1].strip()
                text_en=re.sub(r"\[[^\]]+\]", "", text_en, 0).strip().replace("\n\n", "\n")
                text_target=open(matching_path).read()
                text_target=re.sub(r"^[^,.?!，。？！]+(:|：)", "", text_target, count=1).strip()
                content=text_en+"\n\n"+text_target
                open("matched/{}.{}.en_{}".format(file.split(".")[0], i, target_lang), "w", encoding="utf8").write(content)



if __name__ == "__main__":
    generate_pairs1("Parliament_speeches_htmlFullContent.jsonl")


