import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Loads data from a CSV file and returns reviews and labels.
    Use this function only when you have a CSV file (e.g., imdb_reviews.csv).
    """
    df = pd.read_csv(file_path)
    X = df['review'].values
    y = df['label'].values
    return X, y

def load_imdb_data(data):
    """
    Loads the IMDB dataset from the folder structure (aclImdb) and returns a DataFrame.
    The function expects the base folder (e.g., .../data/aclImdb) as input.
    """
    data_list = []
    
    # Iterate through both 'train' and 'test' splits and both sentiments.
    for split in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            folder_path = os.path.join(data, split, sentiment)
            if not os.path.exists(folder_path):  # Handle missing paths gracefully.
                raise FileNotFoundError(f"❌ Path not found: {folder_path}")
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data_list.append({
                        'review': f.read().strip(),
                        'label': 1 if sentiment == 'pos' else 0
                    })
    
    return pd.DataFrame(data_list)

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into train and test sets and returns two DataFrames.
    Note: This function expects a DataFrame with 'review' and 'label' columns.
    """
    # Extract values from the DataFrame
    X = df['review'].values
    y = df['label'].values
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    
    # Convert the splits back into DataFrames
    train_df = pd.DataFrame({'review': X_train, 'label': y_train})
    test_df = pd.DataFrame({'review': X_test, 'label': y_test})
    
    return train_df, test_df

# -------------------- USAGE SECTION --------------------

# (Optional) If you already have a CSV file with IMDB reviews, you could use:
# csv_file_path = r"D:\feed_ forward_encoder\data\imdb_reviews.csv"
# X, y = load_data(csv_file_path)

# For creating the CSV from the folder structure (aclImdb),
# specify the folder containing the IMDB data.
imdb_folder = r"D:\feed_ forward_encoder\data\aclImdb"

# Load data from the folder structure and convert it to a DataFrame
imdb_df = load_imdb_data(imdb_folder)

# Create the output directory if it doesn't exist.
output_dir = r"D:\feed_ forward_encoder\data"
os.makedirs(output_dir, exist_ok=True)

# Split the DataFrame into train and test sets.
train_df, test_df = split_data(imdb_df)

# Save the DataFrames as CSV files.
train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
imdb_df.to_csv(os.path.join(output_dir, "imdb_reviews.csv"), index=False)

print("✅ IMDB data has been successfully converted to CSV!")
