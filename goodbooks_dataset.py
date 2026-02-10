import numpy as np
import pandas as pd

books = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/refs/heads/master/books.csv')

print(books.iloc[0])