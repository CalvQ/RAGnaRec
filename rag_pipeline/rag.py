import torch
from .data_loader import load_data
from .retrieval import create_retriever
from .generator import initialize_pipeline, generate_response

splits = {
    'train': 'hf://datasets/Yelp/yelp_review_full/train-00000-of-00001.parquet',
    'test': 'hf://datasets/Yelp/yelp_review_full/test-00000-of-00001.parquet'
}

SAMPLE_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COLUMN = "text"

def main(user_review):
    # load data
    df = load_data(splits['train'], splits['test'], sample_size=SAMPLE_SIZE)

    # create retriever
    retriever = create_retriever(
        df,
        content_column=TEXT_COLUMN,
        embedding_model=EMBEDDING_MODEL
    )

    # initialize generation pipeline
    pipe = initialize_pipeline()
    response = generate_response(pipe, user_review, retriever)
    return response

if __name__ == "__main__":
    # Example user review
    user_review = input("Enter a user review: ").strip()
    result = main(user_review)
    print("\nGenerated Response:\n")
    print(result)