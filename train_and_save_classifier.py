import os
import sys
import numpy as np
import pandas as pd
import joblib

# Append the project root to sys.path so that packages like models and utils are found.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models.encoder import EncoderModel
from models.classifier_model import ClassifierModel
from utils.data_preprocessing import load_imdb_data, split_data

# -------------------- Load Training Data --------------------

# Option 1: Load pre-split training data from a CSV file (if available)
train_csv_path = r"D:\feed_ forward_encoder\data\train_data.csv"
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    print("Loaded training data from CSV.")
else:
    # Option 2: Load the complete dataset from the folder structure and split it
    imdb_folder = r"D:\feed_ forward_encoder\data\aclImdb"
    imdb_df = load_imdb_data(imdb_folder)
    train_df, _ = split_data(imdb_df)
    print("Loaded and split training data from folder structure.")

# Extract reviews and labels from the DataFrame
X_train = train_df['review'].values
y_train = train_df['label'].values

# -------------------- Compute Embeddings --------------------

# Instantiate the encoder model
encoder = EncoderModel()

def get_embeddings_in_batches(encoder, texts, batch_size=8):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Ensure batch is a list of strings
        embeddings = encoder.get_embedding(list(batch))
        all_embeddings.append(embeddings)
    return np.concatenate(all_embeddings, axis=0)

# Compute embeddings for training reviews
X_train_embeddings = get_embeddings_in_batches(encoder, list(X_train), batch_size=8)
print("Computed training embeddings.")

# -------------------- Train the Classifier --------------------

# Instantiate your classifier model and train it
classifier = ClassifierModel()
classifier.train(X_train_embeddings, y_train)
print("Classifier trained.")

# -------------------- Save the Trained Classifier --------------------

save_path = r"D:\feed_ forward_encoder\data\saved_classifier.pkl"
joblib.dump(classifier.model, save_path)
print(f"Trained classifier saved to {save_path}")
