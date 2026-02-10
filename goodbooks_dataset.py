import numpy as np
import pandas as pd

books = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/refs/heads/master/books.csv')

print(books.iloc[0])

ratings = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/refs/heads/master/ratings.csv')

print(ratings.iloc[0])

book_tags = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/refs/heads/master/book_tags.csv')

print(book_tags.iloc[0])

to_read = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/refs/heads/master/to_read.csv')

print(to_read.iloc[0])

tags = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/refs/heads/master/tags.csv')

print(tags.iloc[0])
