import torch
from transformers import pipeline

def initialize_pipeline(model_name= "google/gemma-2-2b-it", device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        pipe = pipeline(
            "text-generation",
            model = model_name,
            model_kwargs = {"torch_dtype": torch.bfloat16},
            device = device,
        )
    except Exception as e:
        print(f"Error loading the model: {e}. Falling back to CPU.")
        device = torch.device("cpu")
        pipe = pipeline("text-generation", model = model_name, device = device)
    
    return pipe

def get_template(user_review, context):
    YELP_PROMPT_TEMPLATE = f"""Here are customer reviews similar to a user review:
    BEGIN REVIEWS
    {context}
    END REVIEWS
    Here is the user review: "{user_review}".
    You will act as a recommendation engine. Your task is to analyze the user review and provide thoughtful responses based on the reviews provided. Follow these guidelines:
    1. If the user is reviewing a business:
    - Recognize their positive or negative experience.
    - Highlight 2-3 relevant points that other reviews support.
    - Suggest 1-2 actionable tips or features to enhance their experience.
    2. If the user seeks a recommendation:
    - Respond in a friendly tone, acknowledging their interests or concerns.
    - Recommend 2-3 businesses or features supported by the provided reviews.
    - Briefly explain why each recommendation suits their needs.
    General requirements:
    - Base all statements on the provided reviews. Avoid making unsupported claims.
    - Use phrases like "users often mention..." for summarizing trends.
    - End with a friendly, encouraging closing statement.
    """
    return YELP_PROMPT_TEMPLATE


def get_prompt(user_review, retriever, max_docs = 20):
    retrieved_docs = retriever.get_relevant_documents(user_review)
    if not retrieved_docs:
        return f"No relevant reviews found for the user review: {user_review}"
    
    context = ''
    for i, doc in enumerate(retrieved_docs[:max_docs]):
        context += f"Review {i+1}:\n{doc.page_content}\n"
    return get_template(user_review, context)
    
def generate_response(pipe, user_review, retriever, max_new_tokens = 256):
    prompt = get_prompt(user_review, retriever)
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(messages, max_new_tokens = max_new_tokens)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response

    