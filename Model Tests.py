import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
import keras
from sklearn.naive_bayes import MultinomialNB
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D, Flatten
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
import pickle


def predict_sentiment(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_seq = pad_sequences(text_seq, maxlen=100)

    prediction = model.predict(text_seq)

    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    return sentiment_labels[prediction.argmax()]

# load code borrowed from References listed in ReadMe
model = keras.models.load_model('Sentiment Analysis_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


texts = [
    "I loved this product!",
    "This is the worst product ever!",
    "The product was ok."
]

known_sentiments = ['Positive', 'Negative', 'Neutral']


for i, text in enumerate(texts):
    predicted_sentiment = predict_sentiment(text)
    if predicted_sentiment == known_sentiments[i]:
        print(f"Test {i+1}: Correctly predicted sentiment - {predicted_sentiment}")
    else:
        print(f"Test {i+1}: Incorrectly predicted sentiment - {predicted_sentiment}")