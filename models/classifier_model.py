from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

class ClassifierModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        # setting the iterarion for 1000.
        
    # def train(self, X_train , y_train):
    #     self.model.fit(X_train,y_train)
    def train(self, X_train: 'np.ndarray', y_train: 'np.ndarray') -> None:
        """Train the classifier model on the provided embeddings and labels."""
        self.model.fit(X_train, y_train)

    def predict(self ,X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, y_true, y_pred):
        accuracy = accuracy_score(y_true , y_pred)
        print("accuracy: {accuracy:.4f}")