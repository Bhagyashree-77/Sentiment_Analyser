from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
MODEL_PATH = "../model_save"  # Path to saved model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
model.eval()

# Initialize FastAPI app
app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

@app.post("/analyze/")
def analyze_sentiment(request: SentimentRequest):
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    # Define sentiment labels
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    return {"sentiment": sentiment_labels[prediction], "score": logits.tolist()}

# Run API: uvicorn backend:app --reload
