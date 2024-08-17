import scrapy
import os
from scrapy.crawler import CrawlerProcess
import json


class Spider_google(scrapy.Spider):

    name = 'spider_google'

    allowed_domains = ['sites.google.com']

    start_urls=[
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/buku-sekolah-elektronik/bse-smp-mts/kelas-7",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/buku-sekolah-elektronik/bse-smp-mts/kelas-8",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/buku-sekolah-elektronik/bse-smp-mts/kelas-9",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/buku-sekolah-elektronik/bse-sma-ma/kelas-10",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/buku-sekolah-elektronik/bse-sma-ma/kelas-11",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/buku-sekolah-elektronik/bse-sma-ma/kelas-12",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/belajar-bahasa-asing",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/biografi-tokoh",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/ensiklopedia",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/harun-yahya",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/motivasi",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/novel",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/novel/special-novel-tere-liye",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/novel/special-novel-asma-nadia",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/islami/special-ramadhan",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/islami/terjemahan-kitab",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/islami/quran",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/islami/hadits",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/islami/kumpulan-doa",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/islami/fikih-dan-ibadah",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/komputer-teknologi",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/kamus",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/psikologi",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/psikologi/psikotes",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/pendidikan",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/tips-trik-belajar",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/hobby/budidaya",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/hobby/resep-memasak",
        "https://sites.google.com/madrasah.kemenag.go.id/ysuperpus/pilih-kelompok-buku/sejarah-history",
        ]


    def parse(self, response):


        book_nodes = response.xpath('//a[@class="XqQF9c"]')
        for book_node in book_nodes:
            book_link=book_node.xpath('./@href').get()
            title=book_node.xpath('./text()').get()
            if book_link:
                yield {
                    "title": title,
                    "book_link": book_link
                }

    def warn_on_generator_with_return_value_stub(spider, callable):
        pass

    scrapy.utils.misc.warn_on_generator_with_return_value = warn_on_generator_with_return_value_stub
    scrapy.core.scraper.warn_on_generator_with_return_value = warn_on_generator_with_return_value_stub


def main(output_path='/home/user/data/data_ebook/text_book.id.jsonl'):

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

    process.crawl(Spider_google)
    process.start()


if __name__ == "__main__":
    main()
