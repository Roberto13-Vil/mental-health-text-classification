import numpy as np
import emoji
import re
import os
import string
import nltk
import contractions

nltk_data_path = os.path.join(os.path.dirname(__file__), '../nltk_data')
nltk.data.path.append(nltk_data_path)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags
         "]+", flags=re.UNICODE)

def text_cleaning(text):
    """Clean the input text by removing newlines and tabs, and replace emojis with text placeholders."""
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = emoji.demojize(text) 
    text = text.strip().lower()
    text = re.sub(emoji_pattern, '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_text(text):
    """Apply preprocessing steps to prepare text for classification"""
    text = text_cleaning(text)
    text = contractions.fix(text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[_\W]+', ' ', text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

def vectorize_text(tokens, model):
    """Vectorize the list of tokens using Word2Vec"""
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)