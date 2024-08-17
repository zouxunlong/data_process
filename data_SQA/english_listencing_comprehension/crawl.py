import scrapy
import os
from scrapy.crawler import CrawlerProcess


class Spider_hxen(scrapy.Spider):
    name = 'spider_hxen'
    allowed_domains = ['hxen.com']
    start_urls = [
        'http://www.hxen.com/englishlistening/gaokao/',
    ]

    def parse(self, response):

        articles = response.xpath(
            '//*[@id="content"]//h3[@class="fz18 YaHei fbold"]')
        for article in articles:
            title = article.xpath('./a/text()').get()
            url = article.xpath('./a/@href').get()

            yield response.follow(url=url,
                                  callback=self.parse_article,
                                  cb_kwargs={"title": title})

        next_page_link = response.xpath(
            '//div[@class="pageBar fr "]/a[last()-1]/@href').get()
        if next_page_link:
            yield response.follow(next_page_link, callback=self.parse)

    def parse_article(self, response, *args, **kwargs):

        title = kwargs["title"]
        audio_link = response.xpath(
            '//div[@class="mp3player"]//a[contains(text(), "音频下载")]/@href').get()
        text_nodes = response.xpath('//div[@id="arctext"]/p')
        texts = [''.join(text_node.xpath(".//text()").getall()).replace('\n', " ")
                 for text_node in text_nodes if not text_node.xpath('.//script')]
        text = "\n".join([t.strip() for t in texts if t.strip()]).replace(
            u'\xa0', " ").replace(u'\u3000', " ")
        
        if text and title and audio_link:
            yield {
                "title": title.strip(),
                "text": text.strip(),
                "audio_link": audio_link
            }

    def warn_on_generator_with_return_value_stub(spider, callable):
        pass

    scrapy.utils.misc.warn_on_generator_with_return_value = warn_on_generator_with_return_value_stub
    scrapy.core.scraper.warn_on_generator_with_return_value = warn_on_generator_with_return_value_stub


def main(output_path='/home/user/data/data_SQA/english_listencing_comprehension/hxen_json/hxen.jsonl'):

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

    process.crawl(Spider_hxen)
    process.start()


if __name__ == "__main__":
    main()
