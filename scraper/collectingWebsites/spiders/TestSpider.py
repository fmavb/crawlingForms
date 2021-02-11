from scrapy.spiders import CrawlSpider
from scrapy.http.response.html import HtmlResponse
from bs4 import BeautifulSoup
from scrapy.spiders.crawl import Rule
from scrapy.linkextractors import LinkExtractor
from hashlib import sha256
import scrapy

class TestSpider(CrawlSpider):
    name = "TestSpider"
    start_urls = ['https://www.gov.uk', 'https://www.lesoir.be', 'https://amazon.co.uk']
