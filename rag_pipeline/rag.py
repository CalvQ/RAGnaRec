import torch
from .data_loader import load_data
from .retrieval import create_retriever
from .generator import initialize_pipeline, generate_response
import getpass
from huggingface_hub import login

SAMPLE_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COLUMN = "text"

class RAGPipeline:
    def __init__(self):
        self.df = None
        self.retriever = None
        self.pipe = None
    
    def setup(self):
        huggingface_login()
        self.df = load_data(SAMPLE_SIZE)
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

def main(user_review):
    if not rag_pipeline.pipe:
        rag_pipeline.setup()
    return rag_pipeline.generate_review(user_review)

if __name__ == "__main__":
    # Example user review
    user_review = input("Enter a user review: ").strip()
    result = main(user_review)
    print("\nGenerated Response:\n")
    print(result)