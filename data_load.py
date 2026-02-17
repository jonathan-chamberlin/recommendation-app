import torch as T
import pandas as pd

class MyDataset(T.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        self.length = self.dataset.shape[0]
        
        return self.length
        
    def __getitem__(self, idx):
        
        row = self.dataset.iloc[idx]
        
        user_id = row['user_id']
        book_id = row['book_id']
        rating = row['rating']
        
        return (book_id, user_id, rating)

dataset = pd.read_csv('data/ratings.csv')

ratings_data = MyDataset(dataset)

batch_size = 50
ratings_dataloader = T.utils.data.DataLoader(ratings_data, batch_size, shuffle = False)

