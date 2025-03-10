# Eliminate punctuation
import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
# Download stopwords nltk if it is not already downloaded
nltk.download('stopwords')
# Read data from CSV
df = pd.read_csv('imdb_reviews.csv')
# Look at the first few rows of data to understand its structure
print(df.head())
# Function to clear text
def clean_text(text):
 # Remove punctuation
 text = text.translate(str.maketrans('', '', string.punctuation))
 # Converts text to lowercase
 text = text.lower()
 # Removing stop words
 stop_words = set(stopwords.words('english'))
 text = ' '.join(word for word in text.split() if word not in
stop_words)
 return text
# Apply the clean_text function to the review column
df['Cleaned Review'] = df['review'].apply(clean_text)
df = pd.DataFrame(df)
# Displays the first few rows of cleaned data
df

# Save the cleaned data to CSV
df.to_csv('imdb_cleaned_harry_reviews.csv', index=False)
print('Data cleaned and saved to imdb_cleaned_harry_reviews.csv')