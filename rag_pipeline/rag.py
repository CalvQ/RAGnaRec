import torch
from .data_loader import load_data
from .retrieval import create_retriever
from .generator import initialize_pipeline, generate_response
import getpass
from huggingface_hub import login
import pandas as pd

SAMPLE_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COLUMN = "text"
CSV_FILE_PATH = "sentiment_analysis/sentiment_1000_cached.csv"
TOPIC_COLUMN = "topic"

class RAGPipeline:
    def __init__(self):
        self.df = None
        self.retriever = None
        self.pipe = None
    
    def setup(self):
        huggingface_login()
        self.df = pd.read_csv(CSV_FILE_PATH)
        self.retriever = create_retriever(
            self.df,
            content_column = TEXT_COLUMN,
            embedding_model = EMBEDDING_MODEL
        )
        self.pipe = initialize_pipeline()
    
    def generate_review(self, user_review):
        if not self.pipe or not self.retriever:
            self.setup()
        return generate_response(self.pipe, user_review, self.retriever)

def huggingface_login():
    print("Please enter your Hugging Face token:")
    token = getpass.getpass()
    login(token = token)
    print("Successfully logged in to Hugging Face.")

rag_pipeline = RAGPipeline()

# class RAGPipeline:
#     def __init__(self):
#         self.df = None
#         self.retrievers = {}
#         self.pipe = None

#     def setup(self):
#         huggingface_login()
#         self.df = pd.read_csv(CSV_FILE_PATH)
#         self.pipe = initialize_pipeline()

#     def get_retriever(self, topic_num = None):
#         if topic_num in self.retrievers:
#             return self.retrievers[topic_num]
        
#         retriever = create_retriever(
#             self.df,
#             content_column = TEXT_COLUMN,
#             topic_column = TOPIC_COLUMN,
#             topic_num = topic_num,
#             embedding_model = EMBEDDING_MODEL
#         )
#         self.retrievers[topic_num] = retriever
#         return retriever
    
#     def generate_review(self, user_review, topic_num = None):
#         if not self.pipe:
#             self.setup()
#         retriever = self.get_retriever(topic_num)
#         return generate_response(self.pipe, user_review, retriever)
    
def main(user_review):
    if not rag_pipeline.pipe:
        rag_pipeline.setup()
    return rag_pipeline.generate_review(user_review)

# def main(user_review, topic_num = None):
#     return rag_pipeline.generate_review(user_review, topic_num)

if __name__ == "__main__":
    # Example user review
    user_review = input("Enter a user review: ").strip()
    result = main(user_review)
    print("\nGenerated Response:\n")
    print(result)
