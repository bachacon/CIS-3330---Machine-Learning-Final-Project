import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
from wordcloud import WordCloud
from collections import Counter

## Code used to isentify and clean distribution ##
df = pd.read_csv('Cleaned Sample.csv', low_memory=False)

value_1 = df['Sentiment'].value_counts()['Positive']
value_2 = df['Sentiment'].value_counts()['Neutral']
value_3 = df['Sentiment'].value_counts()['Negative']

print(f"Positive Values: {value_1}")
print(f"Neutral Values: {value_2}")
print(f"Negative Values: {value_3}")
print(df.head())

## Code used to clean distribution ## 
index_rating_pos = df[df['Sentiment'] == 'Positive']
index_rating_pos = index_rating_pos.sample(38000, random_state=1).index
df = df.drop(index_rating_pos, axis=0)

value_1 = df['Sentiment'].value_counts()['Positive']
value_2 = df['Sentiment'].value_counts()['Neutral']
value_3 = df['Sentiment'].value_counts()['Negative']

print(f"Positive Values: {value_1}")
print(f"Neutral Values: {value_2}")
print(f"Negative Values: {value_3}")

#df.to_csv('Cleaned Sample.csv', index=False)

