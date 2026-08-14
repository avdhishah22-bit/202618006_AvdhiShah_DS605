import pandas as pd
import numpy as np

# Load the scraped dataset (produced by books.py -> raw_books_main.csv)
df = pd.read_csv("raw_books.csv")

# Display the first few records
print(df.head())

# Total number of scraped books
print("Total Records:", len(df))

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate UPC values
print("\nDuplicate UPC Values:")
print(df["UPC"].duplicated().sum())

# -----------------------------
# Data Cleaning
# -----------------------------

# Remove leading and trailing spaces
df["title"] = df["title"].str.strip()
df["category"] = df["category"].str.strip()
df["availability"] = df["availability"].str.strip()
df["product_description"] = df["product_description"].fillna("").str.strip()

# Remove duplicate books based on UPC
df = df.drop_duplicates(subset="UPC")

print("\nTotal Records after removing duplicates:", len(df))

# -----------------------------
# Data Transformation
# -----------------------------

# Convert price to float
df["price"] = df["price"].str.replace("£", "", regex=False)
df["price"] = df["price"].astype(float)

# Convert ratings from text to numeric
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["rating"].map(rating_map)

# Extract stock count from availability, e.g. "In stock (19 available)" -> 19
df["stock_count"] = df["availability"].str.extract(r"(\d+)")
df["stock_count"] = df["stock_count"].fillna(0).astype(int)

# Count the number of words in each product description
df["description_word_count"] = (
    df["product_description"]
    .apply(lambda description: len(str(description).split()))
)

# Categorize books by price
df["price_band"] = pd.cut(
    df["price"],
    bins=[0, 20, 40, 100],
    labels=["Low", "Medium", "High"]
)

# Calculate affordability score (rating per unit price)
df["affordability_score"] = (
    df["rating"] / df["price"]
).round(2)

# Recommend books with rating >= 4 and price < 35
df["recommended"] = np.where(
    (df["rating"] >= 4) & (df["price"] < 35),
    "Yes",
    "No"
)

# Display transformed columns
print("\nPreview of cleaned data:")
print(df[[
    "title",
    "price",
    "rating",
    "stock_count",
    "price_band",
    "affordability_score",
    "recommended"
]].head())

# Save cleaned dataset
df.to_csv("raw_books_main.csv", index=False)

print("\nCleaned dataset saved successfully as 'raw_books_main.csv'.")
