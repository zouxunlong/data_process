import requests
import scrapy
import os
from scrapy.crawler import CrawlerProcess
import json


def get_dates(output_path='sittingDate.txt'):
    
    sittingDates=set()
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
    for start in range(0, 3100,20):
        body["startIndex"]=start
        body["endIndex"]=start+20
        results=requests.post(base_url, json=body)
        items=json.loads(results.text)
        for item in items:
            # print(item["sittingDate"], flush=True)
            sittingDates.add(item["sittingDate"])

    for date in sorted(sittingDates):
        open(output_path,"w").write(date+"\n")


class Spider_PD(scrapy.Spider):

    name = 'spider_PD'
    sittingdates=open("sittingDate.txt").readlines()
    allowed_domains = ['sprs.parl.gov.sg']
    start_urls=["https://sprs.parl.gov.sg/search/getHansardReport/?sittingDate={}".format(sittingdate.strip()) for sittingdate in sittingdates]

    def parse(self, response):

        item=json.loads(response.text)
        if "htmlFullContent" in item.keys():
            # print("htmlFullContent", flush=True)
            open("sittingDate_htmlFullContent.txt", "a", encoding="utf8").write(response.url.split("=")[-1]+"\n")
            yield {"sittingDate":response.url.split("=")[-1], "htmlFullContent": item["htmlFullContent"]}
        elif "takesSectionVOList" in item.keys():
            # print("takesSectionVOList", flush=True)
            open("sittingDate_takesSectionVOList.txt", "a", encoding="utf8").write(response.url.split("=")[-1]+"\n")
            yield {"sittingDate":response.url.split("=")[-1], "takesSectionVOList": item["takesSectionVOList"]}
        else:
            print("None", flush=True)
            return

    def warn_on_generator_with_return_value_stub(spider, callable):
        pass

    scrapy.utils.misc.warn_on_generator_with_return_value = warn_on_generator_with_return_value_stub
    scrapy.core.scraper.warn_on_generator_with_return_value = warn_on_generator_with_return_value_stub


def main(output_path='Parliament_speeches.jsonl'):

    os.chdir(os.path.dirname(__file__))

    process = CrawlerProcess(
        settings={
            "FEEDS": {
                output_path: {
                    "format": "jsonlines",
                    "overwrite": True,
                    "encoding": "utf8",
                },
            },
            # "AUTOTHROTTLE_ENABLED": True,
            "DOWNLOAD_DELAY" : 0.5,
            "LOG_LEVEL": "INFO",
            "USER_AGENT": "PostmanRuntime/7.28.4",
        }
    )

    process.crawl(Spider_PD)
    process.start()


if __name__ == "__main__":
    main()
