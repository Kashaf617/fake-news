import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load models

lstm_model = load_model('lstm_model.h5')

# Load TF-IDF vectorizer (if used)
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load tokenizer for LSTM
tokenizer = Tokenizer(num_words=5000)
# (You need to fit the tokenizer on your training data and save it)

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit app
st.title("Fake News Detection")
article = st.text_area("Enter news article here...")

if st.button("Check"):
    cleaned_text = preprocess_text(article)



    # Predict using LSTM
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
    lstm_prediction = (lstm_model.predict(padded_sequence) > 0.5).astype(int)[0][0]


    st.write(f"LSTM Prediction: {'Fake News' if lstm_prediction == 1 else 'Real News'}")
