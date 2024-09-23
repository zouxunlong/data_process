import scrapy
import os
from scrapy.crawler import CrawlerProcess
import json


class Spider_archive(scrapy.Spider):
    name = 'spider_archive'
    allowed_domains = ['archive.org']

    start_urls=[json.loads(line)["url"] for line in open("/home/user/data/data_ebook/archive.ms.jsonl")]

    count=0

    def parse(self, response):

        identifier = response.url.split("/")[-1]

        full_text_node = response.xpath('//a[contains(text(), "FULL TEXT")]')
        if full_text_node:
            full_text_link = full_text_node.xpath("./@href").get()
            yield {
                "identifier": identifier,
                "full_text_link": "https://archive.org"+full_text_link
            }
        else:
            self.count+=1
            print("no full text node: {}".format(self.count), flush=True)


    def warn_on_generator_with_return_value_stub(spider, callable):
        pass

    scrapy.utils.misc.warn_on_generator_with_return_value = warn_on_generator_with_return_value_stub
    scrapy.core.scraper.warn_on_generator_with_return_value = warn_on_generator_with_return_value_stub


def main(output_path='/home/user/data/data_ebook/archive.ms2.jsonl'):

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
            "USER_AGENT": "PostmanRuntime/7.28.4",
        }
    )

    process.crawl(Spider_archive)
    process.start()


if __name__ == "__main__":
    main()
