import pytest
from src.main import Tracer
import pandas as pd

test_books = pd.DataFrame({
    'title': ['Harry Potter','Percy Jackson','Technological Civilization','Ultralearning'],
    'authors':['JK Rowling', 'Rick Riordan','Alex Karp','Scott Young'],
    'ratings_5':[20,30,4,12]
})

algorithm = Tracer(test_books)

def test_recommend() -> None:
    assert algorithm.recommend() == {
            'title':'Percy Jackson',
            'authors': 'Rick Riordan',
            'rating_5': 30
            }
