import scrapy
from scrapy.http.response.html import HtmlResponse
from bs4 import BeautifulSoup
from hashlib import sha256
import os

class CrawlSpider(scrapy.Spider):
    name = 'crawl'
    allowed_domains = ['reed.co.uk']
    start_urls = ['https://www.reed.co.uk/']

    def parse(self, response):
        if isinstance(response, HtmlResponse):
            html = BeautifulSoup(response.body, features="lxml")
            forms = html.find_all("form")
            for form in forms:
                hashForm = sha256(str(form).encode("utf-8")).hexdigest()
                if hashForm not in os.listdir("./forms"):
                    writeToFile = open("./forms/" + hashForm, "w+")
                    writeToFile.write(str(form))
                    writeToFile.close()

        for href in response.xpath('//a/@href').getall():
            yield scrapy.Request(response.urljoin(href), self.parse)
