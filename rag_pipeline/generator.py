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
    YELP_PROMPT_TEMPLATE = f"""Here are customer reviews similar to a user review, accompanied by a sentiment and confidence score:
    BEGIN REVIEWS
    {context}
    END REVIEWS
    Here is the user review: "{user_review}".
    
    Your task is to act as a recommendation engine. Analyze the user review and provide a response based on the sentiment and intent of the user. A sentiment of 0 signifies a negative review, while 1 is positive. Follow these guidelines:

    1. **Identify User Intent:**
        - **Good Reviews, Good Intent**: The user had a positive experience and is looking for similar places to explore.
          - Example: "I had a fantastic experience at Bright Smiles Dental. The staff was friendly, and the cleaning was thorough. Can you recommend other dentists with great service?"
          - **Response**: Recommend 2-3 other places with positive reviews and high confidence, highlighting what users liked about them.

        - **Good Reviews, Bad Intent**: The user had a positive experience but wants to avoid certain issues (e.g., price, location, or other negatives).
          - Example: "The daycare at Little Wonders is great, but I’d like to avoid other centers with long waitlists and inflexible hours. Suggestions of places to avoid?"
          - **Response**: **Focus on negative reviews** that highlight the specific issues (e.g., long waitlists or inflexible hours) that the user wants to avoid. These negative reviews should have **higher confidence** to make solid avoidance recommendations.

        - **Bad Reviews, Good Intent**: The user had a negative experience but is still looking for better places to explore.
          - Example: "The doctor at CarePoint Clinic was dismissive and rushed through my appointment. Can you recommend a more attentive healthcare provider?"
          - **Response**: Recommend 2-3 places with **positive reviews** and **higher confidence**, specifically focusing on places where users had more **attentive service** or better experiences than the negative review.

        - **Bad Reviews, Bad Intent**: The user had a negative experience and wants to avoid similar places.
          - Example: "The food at Bob’s Diner was terrible, and the staff was rude. Can you tell me other places to avoid?"
          - **Response**: **Focus exclusively on negative reviews** of places with **higher confidence**, and recommend 2-3 specific places to **avoid**. Pay particular attention to aspects that users disliked, such as poor food or rude service. Ensure that none of the recommendations are places the user would enjoy.

    2. **Recommendations:**
        - If the user seeks **positive recommendations**, recommend 2-3 places supported by **positive reviews** with higher confidence. Highlight what users like about those places.
        - If the user seeks **to avoid places**, recommend 2-3 specific places to avoid, strictly based on **negative reviews** with **higher confidence**. Focus on aspects that users dislike, ensuring not to recommend anything that the user would enjoy.

    3. **General Requirements:**
        - Do not infer any additional information beyond what is explicitly mentioned in the reviews.
        - Be extremely specific in the responses. Provide **names of places** mentioned in the reviews and avoid generalizations. Do not mention specific review numbers (e.g. "Review 1 mentioned...").
        - Ensure that your response is **direct** and **relevant** to the user's query. Avoid unnecessary elaboration.
        - Respond to the user **directly**, in the first person, unless explicitly stated otherwise.
    """
    return YELP_PROMPT_TEMPLATE


def get_prompt(user_review, retriever, max_docs = 20):
    retrieved_docs = retriever.get_relevant_documents(user_review)
    if not retrieved_docs:
        return f"No relevant reviews found for the user review: {user_review}"
    
    context = ''
    for i, doc in enumerate(retrieved_docs[:max_docs]):
        sentiment = doc.metadata.get('sentiment', 'Unknown')
        confidence = doc.metadata.get('confidence', 'Unknown')
        context += f"Review {i+1}:\n{doc.page_content}\nSentiment: {sentiment}, Confidence: {confidence}\n\n"
    return get_template(user_review, context)
    
def generate_response(pipe, user_review, retriever, max_new_tokens = 256):
    prompt = get_prompt(user_review, retriever)
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(messages, max_new_tokens = max_new_tokens)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response

    