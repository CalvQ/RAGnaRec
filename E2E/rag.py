from retrieval import create_retriever
from generator import initialize_pipeline, generate_response
import getpass
from huggingface_hub import login

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_COLUMN = "text"

class RAGPipeline:
    def __init__(self):
        self.df = None
        self.pipe = None
        self.setup()
    
    def setup(self):
        huggingface_login()
        self.pipe = initialize_pipeline()
    
    def generate_review(self, user_review, cluster_df):
        retriever = create_retriever(
            cluster_df,
            content_column = TEXT_COLUMN,
            embedding_model = EMBEDDING_MODEL
        )
        return generate_response(self.pipe, user_review, retriever)

def huggingface_login():
    print("Please enter your Hugging Face token:")
    token = getpass.getpass()
    login(token = token)
    print("Successfully logged in to Hugging Face.")