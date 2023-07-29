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



## Functions ##
def get_missing_rows(element_df):

    missing_val = element_df.isna()
    rows_missing_val = element_df[missing_val.any(axis=1)]

    return rows_missing_val

def get_missing_columns(element_df):

    missing_val = element_df.isna()
    columns_missing_val = element_df.columns[missing_val.any(axis=0)]

    return columns_missing_val

def preprocess_data(df, column_name):
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.lower()

    df[column_name] = df[column_name].apply(word_tokenize)
    df[column_name] = df[column_name].apply(lambda x: [word for word in x if word not in string.punctuation])

    stop_words = set(stopwords.words('english'))

    df[column_name] = df[column_name].apply(lambda x: [word for word in x if word not in stop_words])


    return df

def get_top_words(words_list):
    word_counts = Counter(words_list)
    top_words = word_counts.most_common(3)
    return [word[0] for word in top_words]




#df = pd.read_csv("Amazon_Reviews_Part1.csv", low_memory=False)
#df = df.sample(100000)
#df = df.to_csv("Amazon Product Reviews 100k Sample1.csv")

df = pd.read_csv("Amazon Reviews 3.csv")
## Combining "Summary" and "Text" Columns ##
#df["Review"] = df['Summary'] + ' ' + df['Text']
# Dropping columns that we do not need
'''
df.drop(['Summary', 'Text', 'Id', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
                  'HelpfulnessDenominator', 'Time', 'ProductId'], axis=1, inplace=True)

# Dropping data for Amazon Reviews Part 3
df.drop(['reviewerName', 'total_vote', 'score_pos_neg_diff',
          'score_average_rating', 'wilson_lower_bound', 'reviewTime', 'day_diff', 'helpful_yes', 'helpful_no'], axis=1, inplace=True)
print(df.head())
df['Score'] = df['overall'].astype(int)
df['Review'] = df['reviewText'].astype(str)
df.drop(['reviewText', 'overall'], axis=1, inplace=True)
print(df.head())
df.to_csv('Amazon Reviews 3.csv', index=False)
'''

# Creating sentiment column code borrowed from references listed in ReadMe
df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 3
                                                    else 'Negative' if x < 3
                                                    else 'Neutral')

# View missing columns and rows and drop missing values #

print(get_missing_columns(df))
print(get_missing_rows(df))
df.dropna()

## Identifying Number of values for each Score ##
value_1 = df['Score'].value_counts()[1]
value_2 = df['Score'].value_counts()[2]
value_3 = df['Score'].value_counts()[3]
value_4 = df['Score'].value_counts()[4]
value_5 = df['Score'].value_counts()[5]

print("Number of Scores of 1:", value_1)
print("Number of Scores of 2:", value_2)
print("Number of Scores of 3:", value_3)
print("Number of Scores of 4:", value_4)
print("Number of Scores of 5:", value_5)
print("\n")

## More distribution cleaning ##
index_rating_5 = df[df['Score'] == 5]
index_rating_5 = index_rating_5.sample(3100, random_state=1).index
df = df.drop(index_rating_5, axis=0)


value_1 = df['Score'].value_counts()[1]
value_2 = df['Score'].value_counts()[2]
value_3 = df['Score'].value_counts()[3]
value_4 = df['Score'].value_counts()[4]
value_5 = df['Score'].value_counts()[5]

## Verifying Value count change
print("Number of Scores of 1:", value_1)
print("Number of Scores of 2:", value_2)
print("Number of Scores of 3:", value_3)
print("Number of Scores of 4:", value_4)
print("Number of Scores of 5:", value_5)

print(df.head())
#df.to_csv("Cleaned Sample3.csv", index=False)
