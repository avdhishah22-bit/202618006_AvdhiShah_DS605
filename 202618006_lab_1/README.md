# Lab 1 – Data Scraping and Preprocessing

**Course:** DS605
**Name:** Avdhi Shah
**ID:** 202618006

## Project Overview

The folder contains a complete data pipeline around [books.toscrape.com](https://books.toscrape.com/):
it starts with scraping book data with a Scrapy spider, then cleaning and transforming it with pandas, followed by generating visualizations and a word cloud, and drawing data-driven insights from the results.

The pipeline:

1. **Scrape** — `books.py` crawls the catalogue as well as each individual book page, it extracts
   title, category, price, rating, availability, product description, UPC, number of reviews,
   and product URL for 100 books across 5 catalog pages. it gives: `raw_books.csv`.
2. **Clean & Transform** — `cleaning.py` preprocess the data like removeing duplicates, converting price
   and rating to numeric types, extracting stock counts from the availability text, as well as engineers
   new features (description_word_count, price_band, affordability_score, recommended).
   it gives: `raw_books_main.csv`.
3. **Visualize & Analyze** — `visuals.py` produces price/rating distributions, average price by
   category, a price-vs-rating scatter plot, and a word cloud from the book descriptions, along
   with summary statistics and category-level breakdowns.

`cleaningvisuals.ipynb` combines steps 2 and 3 into a single notebook for an end-to-end,
cell-by-cell walkthrough with inline outputs and plots.



## Results 

- 100 books scraped from 5 catalog pages.
- Prices range from £10.16 to £58.11 (mean £34.56), split into Low/Medium/High price bands.
- Price and rating show no meaningful correlation (r ≈ -0.12) — price is not a signal of
  perceived quality in this dataset.
- Sequential Art and Nonfiction are the most-represented categories by book count, while
  Childrens, Fiction, and Music are the most expensive on average.
- 21 of 100 books meet the `recommended` criteria (rating ≥ 4 and price < £35).
- Full write-up of observations, category patterns, and best-value picks is in the
  "Insights and Interpretation" section at the end of `cleaning_visuals.ipynb`.



