from fastapi import FastAPI
from main import Tracer
import pandas as pd
from tests.test_tracer import test_books

model = Tracer(books = test_books)

# pd.read_csv('data/books.csv')

app = FastAPI()

@app.get("/recommend")

def get_tracer_recommendation():
    
    recommendation = model.recommend()
    
    recommendation['rating_5'] = int(recommendation['rating_5'])
    #required because FastAPI doesn't know how to convert numpy.int64 to json
    
    return recommendation