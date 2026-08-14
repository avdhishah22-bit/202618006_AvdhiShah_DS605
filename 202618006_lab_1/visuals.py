import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Apply plot style
sns.set_theme(style="whitegrid")

# Read the cleaned dataset (produced by cleaning.py)
books = pd.read_csv("raw_books_main.csv")

print(books.head())

# ------------------------------------------
# Price Distribution
# ------------------------------------------

plt.figure(figsize=(8, 5))
plt.hist(books["price"], bins=15, edgecolor="black")
plt.title("Book Price Distribution")
plt.xlabel("Price (£)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plot_price_distribution.png", dpi=150)
plt.show()

# ------------------------------------------
# Rating Distribution
# ------------------------------------------

plt.figure(figsize=(8, 5))
sns.countplot(x="rating", data=books, order=sorted(books["rating"].unique()))
plt.title("Book Rating Distribution")
plt.xlabel("Rating (stars)")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.savefig("plot_rating_distribution.png", dpi=150)
plt.show()

# ------------------------------------------
# Average Price in Each Category
# ------------------------------------------

category_prices = books.groupby("category")["price"].mean()
category_prices = category_prices.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(category_prices.index, category_prices.values)
plt.xticks(rotation=45, ha="right")
plt.title("Average Price by Category")
plt.xlabel("Category")
plt.ylabel("Average Price (£)")
plt.tight_layout()
plt.savefig("plot_avg_price_by_category.png", dpi=150)
plt.show()

# ------------------------------------------
# Price and Rating Comparison
# ------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(books["price"], books["rating"], alpha=0.7)
plt.title("Price vs Rating")
plt.xlabel("Price (£)")
plt.ylabel("Rating")
plt.tight_layout()
plt.savefig("plot_price_vs_rating.png", dpi=150)
plt.show()

# ------------------------------------------
# Word Cloud
# ------------------------------------------

all_descriptions = " ".join(books["product_description"].fillna(""))

cloud = WordCloud(width=900, height=450, background_color="white")
cloud.generate(all_descriptions)

plt.figure(figsize=(12, 6))
plt.imshow(cloud)
plt.axis("off")
plt.title("Common Words in Book Descriptions")
plt.tight_layout()
plt.savefig("plot_wordcloud.png", dpi=150)
plt.show()

# ------------------------------------------
# Dataset Summary
# ------------------------------------------

print("\nSummary Statistics\n")
print(books.describe(include="all"))

# ------------------------------------------
# Highest Rated Books
# ------------------------------------------

best_books = books.sort_values(["rating", "price"], ascending=[False, True])

print("\nTop Rated Books\n")
print(best_books[["title", "category", "rating", "price"]].head(10))

# ------------------------------------------
# Category Statistics
# ------------------------------------------

category_stats = books.groupby("category").agg(
    Average_Price=("price", "mean"),
    Average_Rating=("rating", "mean"),
    Total_Stock=("stock_count", "sum")
)

category_stats = category_stats.round(2)

print("\nCategory Summary\n")
print(category_stats)
