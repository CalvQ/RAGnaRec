from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader

def get_topic(user_review):
    # add topic detection logic - TO DO
    return 0

def create_retriever(
        df, content_column = "text", chunk_size = 1000, chunk_overlap = 200,
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2", search_type = "similarity"):
    loader = DataFrameLoader(df, page_content_column = content_column) # Replace with topic documents
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name = embedding_model)
    vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings)

    return vectorstore.as_retriever(search_type = search_type)


def retrieve_similar_documents(user_review,
                               df, content_column = "text", chunk_size = 1000, chunk_overlap = 200,
                               embedding_model = "sentence-transformers/all-MiniLM-L6-v2", search_type = "similarity"):
    topic = get_topic(user_review) # Not used yet, TO DO
    retriever = create_retriever(
        df, content_column = content_column, chunk_size = chunk_size, chunk_overlap = chunk_overlap, embedding_model = embedding_model, search_type = search_type)
    similar_docs = retriever.get_relevant_documents(user_review)

    return similar_docs
