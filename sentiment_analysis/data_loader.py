import pandas as pd
import random
from datasets import load_dataset

def load_data(sample_size = 1000):
    #splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet', 'test': 'yelp_review_full/test-00000-of-00001.parquet'}
    dataset = load_dataset("yelp_review_full")
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])
    df = pd.concat([df_train, df_test], ignore_index=True)

    if sample_size:
        random.seed(10701)
        indices = random.sample(range(len(df)), 1000)
        df = df.iloc[indices]

    return df