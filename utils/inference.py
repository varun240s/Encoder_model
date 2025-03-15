import os, sys
import pandas as pd
import numpy as np
import joblib
# from utils.inference import predict_sentiment, predict_sentiments
# Append the project root to sys.path so that the models package is found.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoder import EncoderModel
from models.classifier_model import ClassifierModel

def predict_sentiment(text, encoder, classifier):
    """
    Predict the sentiment for a single review.
    """
    # Convert the single review text into a list.
    text_list = [text]
    embedding = encoder.get_embedding(text_list)
    prediction = classifier.predict(embedding)[0]
    return "Positive" if prediction == 1 else "Negative"

def predict_sentiments(texts, encoder, classifier, batch_size=8):
    """
    Predict sentiments for a batch of reviews.
    """
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = encoder.get_embedding(batch)
        batch_prediction = classifier.predict(embeddings)
        predictions.extend(["Positive" if pred == 1 else "Negative" for pred in batch_prediction])
    return predictions



if __name__ == "__main__":
    # Instantiate the encoder model
    encoder = EncoderModel()
    
    # Instantiate the classifier and load the trained model weights
    classifier = ClassifierModel()
    model_path = r"D:\feed_ forward_encoder\data\saved_classifier.pkl"
    if os.path.exists(model_path):
        classifier.model = joblib.load(model_path)
        print("✅ Loaded trained classifier from disk.")
    else:
        raise FileNotFoundError(f"Trained classifier not found at {model_path}. Please train and save your model first.")
    
    # Test with a single review
    sample_review = "I absolutely loved this movie! but public really hated it."
    single_prediction = predict_sentiment(sample_review, encoder, classifier)
    
    print(f"review : {sample_review}\n Single Review Prediction:", single_prediction)
    print("----------------------------@----------------------------")
    # Test with a batch of reviews
    batch_reviews = [
        "This movie is good but i don't like it.",
        "An outstanding performance by the lead actor!",
        "Not my type of film, pretty boring.",
        "An absolute masterpiece."
    ]
    batch_predictions = predict_sentiments(batch_reviews, encoder, classifier, batch_size=8)
    for review, pred in zip(batch_reviews, batch_predictions):
        print(f"Review: {review}\nPredicted Sentiment: {pred}\n")
