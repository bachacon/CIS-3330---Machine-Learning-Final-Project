import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

## Functions
def plot_score_distriution(df, column):
    sns.countplot(data=df, x=df[column])

    plt.title(f"Distribution of {column}")
    plt.show()

def plot_sentiment_distribution(df, column):
    sns.countplot(data=df, x=df[column])
    plt.title(f"Distribution of {column}")
    plt.show()


def get_word_cloud(df, column_name, stopwords=None):
    text = ' '.join(df[column_name].astype(str).tolist())
    wc = WordCloud(width=1000, height=500, stopwords=stopwords, background_color='black').generate(text)

    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def get_text_length(df, text_column, name_of_length_column):
    df[text_column] = df[text_column].astype(str)
    df[name_of_length_column] = df[text_column].apply(len)

    plt.hist(df[name_of_length_column], bins=50)
    plt.xlabel('Text Length')
    plt.ylabel('Count')
    plt.title("Distribution of Review Text Length")
    plt.show()

    mean_length = df[name_of_length_column].mean()
    std_length = df[name_of_length_column].std()

    print(f"Mean text length: {mean_length:.2f}")
    print(f"Standard deviation of text length: {std_length:.2f}")


## Load DF's
cleaned_df = pd.read_csv("Cleaned Sample.csv", low_memory=False)

## General Analysis
print(cleaned_df['Score'].describe())
print("\n")
print("Mean of score:", cleaned_df['Score'].mean())
print("Median of score:", cleaned_df['Score'].median())



# WordCloud
get_word_cloud(cleaned_df, 'Review')

## Score Distribution Analysis
plot_score_distriution(cleaned_df, 'Score')
plot_sentiment_distribution(cleaned_df, 'Sentiment')

## Text Length Analysis
get_text_length(cleaned_df, 'Review', 'Text Length')