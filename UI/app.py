import streamlit as st
import requests

# Streamlit App UI
st.title("Sentiment Analysis with Transformers")
st.write("Enter a sentence to analyze its sentiment.")

# Text Input
user_input = st.text_area("Enter text:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        response = requests.post("http://127.0.0.1:8000/analyze/", json={"text": user_input})
        result = response.json()
        
        # Display sentiment
        st.write(f"**Sentiment:** {result['sentiment']}")
        
        # Display confidence scores (optional)
        st.write(f"**Confidence Scores:** {result['score']}")
    else:
        st.warning("Please enter some text before analyzing.")
## uvicorn backend:app --host 0.0.0.0 --port 8000 --reload