from bertopic import BERTopic
import json
import pandas as pd

class ClusteringModel():
    def __init__(self):
        self.model_path = "../clustering/BERTopic/model"
        self.topic_model = BERTopic.load(self.model_path, embedding_model="all-MiniLM-L6-v2")
        
    def find_topic(self, review):
        # Load topics dictionary
        with open(f"{self.model_path}/topics.json", 'r') as f:
            topics_dict = json.load(f)
        
        topics_dict = topics_dict['topic_representations']

        # Convert review text to lowercase and split into words
        review_words = set(review.lower().split())
        
        max_probability = 0
        best_topic = None
        best_word = None
        best_prob = 0
        
        # For each topic and its list of [word, probability] pairs
        for topic_id, word_pairs in topics_dict.items():
            # Extract words and probabilities for this topic
            for word, prob in word_pairs:
                if word.lower() in review_words and prob > max_probability:
                    max_probability = prob
                    best_topic = int(topic_id)
                    best_word = word
                    best_prob = prob
        
        if best_topic is not None:
            return best_topic
            # return best_topic, (best_word, best_prob)
        return -1
        # return -1, None

        
    def assign_topic(self, review):
        topic, _ = self.topic_model.transform([review])
        if topic == -1:
            return self.find_topic(review)
        return topic[0]
        # return {
        #     'topic': topics[0],
        #     'probability': probabilities[0],
        #     'keywords': self.topic_model.get_topic(topics[0]) if topics[0] != -1 else None
        # }
        
    def fetch_cluster(self, topic):
        if topic == -1:
            print("Please input a more specific review. We are not sure what to recommend based on this input.")
        return pd.read_csv(f"../clustering/clusters/topic{topic}.csv")
        
    