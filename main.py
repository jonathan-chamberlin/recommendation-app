print('hello')
import pandas as pd
class Tracer:
    def __init__(self, books):
        self.books = books
    
    def recommend(self):
        # in books, find row where 'rating_5' is max
        
        rating_5_series = self.books['ratings_5']
        
        index_max = rating_5_series.idxmax()
        
        row = self.books.loc[index_max]
        
        title = row['title']
        author = row['authors']
        rating_5 = row['ratings_5']
        
        return {
            'title':title,
            'authors': author,
            'rating_5': rating_5
            }
