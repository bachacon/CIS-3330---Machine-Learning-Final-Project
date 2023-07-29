import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
from keras.preprocessing.text import Tokenizer
import re
## Functions ##
def preprocess_data(df, column_name):
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.lower()

    df[column_name] = df[column_name].apply(word_tokenize)
    df[column_name] = df[column_name].apply(lambda x: [word for word in x if word not in string.punctuation])

    stop_words = set(stopwords.words('english'))

    df[column_name] = df[column_name].apply(lambda x: [word for word in x if word not in stop_words])


    return df

## This is where I transformed data and preprocessed some data ##

df = pd.read_csv('Transformed Sample.csv')
#preprocess_data(df, 'Review')
print(df.head())
#df.to_csv("Transformed Sample.csv", index=False)
#df['Review'] = df['Review'].astype(str)
#df['Review'] = df['Review'].str.lower()
#print(df.head())
#df.to_csv("Cleaned Sample.csv", index=False)
df_df = pd.read_csv('Cleaned Sample.csv', low_memory=False)
print(df_df.head())
