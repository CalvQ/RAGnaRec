import pandas as pd
import random

def load_data(sample_size = 1000):
    splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet', 'test': 'yelp_review_full/test-00000-of-00001.parquet'}
    df_train = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits["train"])
    df_test = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits["test"])
    df = pd.concat([df_train, df_test], ignore_index=True)

    if sample_size:
        random.seed(10701)
        indices = random.sample(range(len(df)), 1000)
        df = df.iloc[indices]

    return df