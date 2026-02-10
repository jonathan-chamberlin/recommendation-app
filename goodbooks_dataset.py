import numpy as np
import pandas as pd

books = pd.read_csv('data/books.csv')

print(books.iloc[0])

ratings = pd.read_csv('data/ratings.csv')

print(ratings.iloc[0])

book_tags = pd.read_csv('data/book_tags.csv')

print(book_tags.iloc[0])

to_read = pd.read_csv('data/to_read.csv')

print(to_read.iloc[0])

tags = pd.read_csv('data/tags.csv')

print(tags.iloc[0])
