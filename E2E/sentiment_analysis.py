from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

class SentimentAnalyzer:
    def __init__(self, model_name = 'distilbert-base-uncased-finetuned-sst-2-english'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_pipeline = pipeline(
            'sentiment-analysis',
            model = model_name,
            tokenizer = model_name,
            device = device
        )

    def analyze_sentiment(self, text):
        result = self.sentiment_pipeline(text, truncation = True, max_length = 512)[0]
        label_map = {
            'NEGATIVE': 0,
            'POSITIVE': 1
        }

        translated_label = label_map[result['label']]
        return translated_label, result['score']
    
    def analyze_dataframe(self, df, text_column = 'text'):
        df[['sentiment', 'confidence']] = df[text_column].apply(
            lambda x: pd.Series(self.analyze_sentiment(x))
        )
        return df
    
    def evaluate_model(self, df):
        sentiment_counts = df['sentiment'].value_counts()
        average_confidence = df.groupby('sentiment')['confidence'].mean()
        average_label = df.groupby('sentiment')['label'].mean()
        return sentiment_counts, average_confidence, average_label
    