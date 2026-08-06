import pandas as pd
df = pd.read_csv("C:\Users\avdhi\OneDrive\Desktop\202618006_AvdhiShah_DS605\bookscraper\raw_books.csv")
#display output
print(df.head())
#counting total scraped books
print("Total records:", len(df))
#filtering out books with missing values
print("Missing Values:")
print(df.isnull().sum())