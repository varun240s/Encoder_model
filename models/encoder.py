
from transformers import AutoTokenizer, AutoModel
import torch

# Enable CUDA (GPU) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class EncoderModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(device)  # Move model to GPU
        
    def get_embedding(self, texts):
        """Converts text into embeddings using the pretrained model."""
        
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        # Move tensors back to CPU before converting to NumPy
        return output.last_hidden_state[:, 0, :].cpu().numpy()

