from nltk.util import pr
import pandas as pd
import numpy as np
import en_core_web_sm
import re
import spacy
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import joblib

tweet = pd.read_csv("twitter.csv")
#print(tweet.head())

tweet["labels"] = tweet["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#print(tweet.head())

tweet = tweet[["tweet", "labels"]]
#print(tweet.head())

import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean():
    tweet = str().lower()
    tweet = re.sub('\[.*?\]', '', tweet)
    tweet = re.sub('https?://\S+|www\.\S+', '', tweet)
    tweet = re.sub('<.*?>+', '', tweet)
    tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
    tweet = re.sub('\n', '', tweet)
    tweet = re.sub('\w*\d\w*', '', tweet)
    tweet = [word for word in tweet.split(' ') if word not in stopword]
    tweet =" ".join(tweet)
    tweet = [stemmer.stem(word) for word in tweet.split(' ')]
    tweet =" ".join(tweet)
    return tweet

#print(tweet.head())

X = np.array(tweet["tweet"])
y = np.array(tweet["labels"])

if __name__ == '__main__':
    tweet = clean()

    #for row in tweet["tweet"]:

        #print(row)

# Splits the dataset.
class TweetClassfier:
    def __init__(self):
        self.tweet = clean()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # reads the pipeline's config, loads model data and returns it.
        self.nlp = en_core_web_sm.load()
        self.punctiations = string.punctuation  # Give all sets of punctuation.
        self.stop_words = STOP_WORDS  # Remove words that add no meaning to a sentence.
        self.parser = English()
        # convert text to a vectors.
        self.bow_vector = CountVectorizer(tokenizer=self.tokenizer, ngram_range=(1, 1))


        self.pipe = None

        # Separating text into tokens.

    def tokenizer(self, tweet):
        tokens = self.nlp(tweet)

        # Lemmatizing.
        tokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]

        # Remove stop words.
        tokens = [word for word in tokens if word not in self.stop_words and word not in self.punctiations]

        return tokens

    # takes the arguments and returns the ratio of the number of correct predictions.

    def fit(self):
        tw_class = LogisticRegression(max_iter=5000)
        self.pipe = Pipeline([('cleaner', Cleaner()),
                         ('vectorizer', self.bow_vector),
                         ('classifier', tw_class)])

        print('Data fitting. Please wait for a few moments.')
        self.pipe.fit(self.X_train, self.y_train)

    def predict(self):
        predictions = self.pipe.predict(self.X_test)
        print("Logistic Regression Accuracy:", metrics.accuracy_score(self.y_test, predictions))

    def predict_new(self, new_tweet):
        prediction = self.pipe.predict([new_tweet])

        print('Processing tweet: \"{}\"\n\t Classification: {}'.format(new_tweet, prediction))

        return prediction


class Cleaner(TransformerMixin):
    """ Custom transformer using SpaCy.
    """
    # Apply the component’s model
    def transform(self, X, **transform_params):
        return [self.clean(tweet) for tweet in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

    def clean(self, tweet):
        return tweet.strip().lower()


if __name__ == '__main__':
    tc = TweetClassfier()
    tc.fit()
    tc.predict()
