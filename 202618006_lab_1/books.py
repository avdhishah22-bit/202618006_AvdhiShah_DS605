import scrapy


class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]

    page_count = 1
    max_pages = 5

    # Output raw scraped records to visuals.csv automatically when running
    # `scrapy crawl books` (no need to pass -O on the command line).
    custom_settings = {
        "FEEDS": {
            "raw_books.csv": {
                "format": "csv",
                "overwrite": True,
            },
        },
    }

    def parse(self, response):

        # Visit each book page
        for book in response.css("article.product_pod h3 a"):
            relative_url = book.attrib["href"]
            yield response.follow(relative_url, callback=self.parse_book)

        # Follow pagination
        next_page = response.css("li.next a::attr(href)").get()

        if next_page and self.page_count < self.max_pages:
            self.page_count += 1
            yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):

        table = response.css("table.table.table-striped tr")

        data = {}

        for row in table:
            key = row.css("th::text").get()
            value = row.css("td::text").get()
            data[key] = value

        # Category: grab the LAST breadcrumb link (Home > Category > Title,
        # where Title has no <a>), which is more reliable than a fixed
        # nth-child position.
        breadcrumb_links = response.css("ul.breadcrumb li a::text").getall()
        category = breadcrumb_links[-1].strip() if breadcrumb_links else None

        # Availability: join ALL text nodes under p.availability and collapse
        # whitespace, so we reliably capture the full string, e.g.
        # "In stock (19 available)" instead of just "In stock".
        availability_text = " ".join(
            response.css("p.availability::text").getall()
        )
        availability_text = " ".join(availability_text.split())

        yield {
            "title": response.css("div.product_main h1::text").get(),

            "category": category,

            "price": response.css(
                "p.price_color::text"
            ).get(),

            "rating": response.css(
                "p.star-rating"
            ).attrib["class"].split()[-1],

            "availability": availability_text,

            "product_description": response.css(
                "#product_description + p::text"
            ).get(),

            "UPC": data.get("UPC"),

            "number_of_reviews": data.get("Number of reviews"),

            "product_url": response.url
        }
