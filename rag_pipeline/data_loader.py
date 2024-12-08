import pandas as pd
import random

def load_data(train_path, test_path, sample_size = 1000, seed = 10701):
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    df = pd.concat([df_train, df_test], ignore_index = True)

    if sample_size:
        random.seed(seed)
        indices = random.sample(range(len(df)), sample_size)
        df = df.iloc[indices]

    return df