# Import libraries
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# Read cleaned data from CSV
df = pd.read_csv('imdb_cleaned_harry_reviews.csv')

# Initialize tokenizer and IndoBERT model
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3) # 3 labels: positive, neutral, negative

# Function to carry out sentiment labeling
def predict_sentiment(text):
    # Text tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # Make predictions using models
    with torch.no_grad():
        outputs = model(**inputs)
    # Calculating probabilities
    probs = softmax(outputs.logits, dim=-1)
    # Get the label with the highest probability
    label = torch.argmax(probs, dim=1).item()
    return label

# Apply sentiment labeling function to cleaned review column
tqdm.pandas() # Untuk progres bar
df['Sentiment'] = df['Cleaned Review'].progress_apply(predict_sentiment)

# Mapping label numbers to text sentiment
sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
df['Sentiment Label'] = df['Sentiment'].map(sentiment_mapping)

# Displays the first few rows of labeled data
print(df.head())

# Save the labeled data to a new CSV file
df.to_csv('imdb_labeled_harry_reviews.csv', index=False)
print('Data labeled and saved to imdb_labeled_harry_reviews.csv')
df