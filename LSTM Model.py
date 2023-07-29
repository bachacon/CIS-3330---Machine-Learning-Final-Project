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

## DataFrame Creation for each CSV ##
df = pd.read_csv("Cleaned Sample.csv")
df['Review'] = df['Review'].astype(str)

## Convert to sequence of integers ##
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Review'])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(df['Review'])
pad_seq = pad_sequences(sequences, maxlen=100, truncating='post')

sentiment_labels = pd.get_dummies(df['Sentiment']).values

# Split the dataset #
x_train, x_test, y_train, y_test = train_test_split(pad_seq, sentiment_labels, test_size=0.2)


## Create Network ##
model = Sequential()
model.add(Embedding(len(word_index)+1, 100, input_length=100, trainable=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
## Compile ##
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

## Train Network #
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


## Evaluation ##
y_prediction = np.argmax(model.predict(x_test), axis=-1)
y_true = np.argmax(y_test, axis=-1)
accuracy = accuracy_score(y_true, y_prediction)
precision = precision_score(y_true, y_prediction, average='macro')
recall = recall_score(y_true, y_prediction, average='macro')
f1 = f1_score(y_true, y_prediction, average='macro')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1_Score:", f1)

# Code used to save model should you run it again borrowed from References #
'''
# Save model #
model.save('sentiment_analysis_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
### Code is borrowed from Tensorflow referenced in README ###
## Model Chart Plots ##
plt.plot(hist.history['accuracy'], label='acc')
plt.plot(hist.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

plt.savefig("Accuracy Plot.jpg")

plt.plot(hist.history['loss'], label='Loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.savefig("Loss Plot.jpg")