


from models.encoder import EncoderModel
from models.classifier_model import ClassifierModel
from utils.data_preprocessing import load_data, split_data
from utils.evaluation import evaluate_model
from sklearn.model_selection import train_test_split

# Use the correct CSV file path here
X, y = load_data(r"D:\feed_ forward_encoder\data\imdb_reviews.csv")

# Train-test split
X_train, X_test, y_train, y_test = split_data(X, y)

# Get embeddings using the encoder
encoder = EncoderModel()
X_train_embeddings = encoder.get_embeddings(X_train)
X_test_embeddings = encoder.get_embeddings(X_test)

# Train the classifier
classifier = ClassifierModel()
classifier.train(X_train_embeddings, y_train)

# Get predictions
y_pred = classifier.predict(X_test_embeddings)

# Evaluate the model
evaluate_model(y_test, y_pred)




























# from models.encoder import EncoderModel
# from models.classifier_model import ClassifierModel
# from utils.data_preprocessing import load_imdb_data, split_data  # use load_imdb_data
# from utils.evaluation import evaluate_model
# from sklearn.model_selection import train_test_split
# import numpy as np

# # new added after traning
# from utils.inference import predict_sentiment, predict_sentiments


# # Use load_imdb_data to get a DataFrame from folder structure
# imdb_df = load_imdb_data(r"D:\feed_ forward_encoder\data\aclImdb")

# # Split the dataframe into train and test sets
# train_df, test_df = split_data(imdb_df)

# # Convert DataFrame back to arrays if needed:
# X_train = train_df['review'].values
# y_train = train_df['label'].values
# X_test = test_df['review'].values
# y_test = test_df['label'].values

# # Loading encoder to get the embeddings
# encoder = EncoderModel()


# def get_embeddings_in_batches(encoder, texts, batch_size=8):
#     all_embeddings = []
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i+batch_size]
#         embeddings = encoder.get_embedding(batch)
#         all_embeddings.append(embeddings)
#     return np.concatenate(all_embeddings, axis=0)

# # Use the new function:
# X_train_embeddings = get_embeddings_in_batches(encoder, list(X_train), batch_size=8)
# X_test_embeddings = get_embeddings_in_batches(encoder, list(X_test), batch_size=8)


# # X_train_embeddings = encoder.get_embedding(list(X_train))
# # X_test_embeddings = encoder.get_embedding(list(X_test))

# # Train the classifier
# classifier = ClassifierModel()
# classifier.train(X_train_embeddings, y_train)

# # Get predictions
# y_pred = classifier.predict(X_test_embeddings)

# # Evaluate
# evaluate_model(y_test, y_pred)



# # new added after traning
# # Test with a single review
# sample_review = "I absolutely loved this movie! It was fantastic and heartwarming."
# prediction = predict_sentiment(sample_review, encoder, classifier)
# print("Single Review Prediction:", prediction)


# # Test with a batch of reviews
# batch_reviews = [
#     "This movie was a complete waste of time.",
#     "An outstanding performance by the lead actor!",
#     "Not my type of film, pretty boring.",
#     "An absolute masterpiece."
# ]
# batch_predictions = predict_sentiments(batch_reviews, encoder, classifier)
# for review, pred in zip(batch_reviews, batch_predictions):
#     print(f"Review: {review}\nPredicted Sentiment: {pred}\n")



