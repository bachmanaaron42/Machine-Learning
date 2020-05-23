import pandas as pd
submissions = pd.read_csv("sel_hn_stories.csv")
submissions.columns = ["submission_time", "upvotes", "url", "headline"]
submissions = submissions.dropna()

# tokenize - split the data into individual list of words
tokenized_headlines = []

for hl in submissions['headline']:
    tokens = hl.split()
    tokenized_headlines.append(tokens)

# remove punctuation and make all tokens lower case
punctuation = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")"]
clean_tokenized = []

for tks in tokenized_headlines:
    tk_headline = []
    for tk in tks:
        clean_tk = tk.lower()
        for p in punctuation:
            clean_tk = clean_tk.replace(p, '')
        
        tk_headline.append(clean_tk)
    
    clean_tokenized.append(tk_headline)

# create list of unique tokens that appear more than once in the text data

import numpy as np
unique_tokens = []
single_tokens = []

count_tokens = {}

for ct in clean_tokenized:
    for tkn in ct:
        if tkn in count_tokens:
            count_tokens[tkn] += 1
        else:
            count_tokens[tkn] = 1
            
for key, value in count_tokens.items():
    if value > 1:
        unique_tokens.append(value)
    else:
        single_tokens.append(value)
        
# create dataframe with columns for each word that appears more than once, and rows for each headline, value is all zeros
counts = pd.DataFrame(0, index=np.arange(len(clean_tokenized)), columns=unique_tokens)

# increment each token for each headline within the counts Dataframe

for index, tk_sentence in enumerate(clean_tokenized):
    for token in tk_sentence:
        if token in unique_tokens:
            counts.iloc[index][token] += 1

# remove words that are used too little or too much to reduce noise and remove stopwords
word_counts = counts.sum(axis=0)

counts = counts.loc[:, (word_counts >= 5) & (word_counts <= 100)]

# split the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(counts, submissions["upvotes"], test_size=0.2, random_state=1)

# use Linear Regression to predict upvotes, based on the bag of words (word count)

from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# calculate mean squared error

diff = predictions - y_test
squares = diff ** 2
total = sum(squares)
mse = total / len(predictions)