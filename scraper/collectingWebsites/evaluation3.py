import os
import pandas as pd
import numpy as np
import mysql.connector
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import xx_ent_wiki_sm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB


class ItemSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, key):
        self.key = key
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, data_dict):
        return data_dict[self.key]

#@Tokenize
def spacy_tokenize(string):
    tokens = list()
    doc = nlp(string)
    for token in doc:
        tokens.append(token)
    return tokens

#@Normalize
def normalize(tokens):
    normalized_tokens = list()
    for token in tokens:
        normalized = token.text.lower().strip()
        if ((token.is_alpha or token.is_digit)):
            normalized_tokens.append(normalized)
    return normalized_tokens

#@Tokenize and normalize
def tokenize_normalize(string):
    return normalize(spacy_tokenize(string))


def predictSensitivity(fields, vectorizer, model, cursor):
    sqlNB = "UPDATE WordNetEval6 SET Bayes=%s WHERE id=%s"
    
    vectors = vectorizer.transform(fields["words"])

    results = model.predict(vectors.toarray())
    
    toDB = []

    for i in range(len(results)):
        toDB.append((results[i].item(), fields['id'][i].item()))
        print("Row %s out of %s rows" % (i, len(results)))

    cursor.executemany(sqlNB, toDB)

    return

db = mysql.connector.connect(host="localhost",
user="root", password="yourpassword", database="ontology")

cursor = db.cursor()

cursor.execute("set @@sql_mode='NO_ENGINE_SUBSTITUTION';")
db.commit()

sql = "SELECT word, sensitivity FROM vocabulary WHERE sensitivity IS NOT NULL GROUP BY word;"
cursor.execute(sql)
result = cursor.fetchall()

sql2 = 'SELECT word, sensitivity FROM vocabularyopencyc WHERE sensitivity IS NOT NULL GROUP BY word'
cursor.execute(sql2)
result += cursor.fetchall()

words = pd.DataFrame(data=result, columns=("words", "sensitivity"))
words = words.sample(frac=1, random_state=2)

training = 0.6
validation = 0.75
testing = 0.8

training_data = words.iloc[:int(len(words)*training),:]
validation_data = words.iloc[int(len(words)*training):int(len(words)*validation),:]
testing_data = words.iloc[int(len(words)*validation):,:]

train_labels = training_data["sensitivity"]
validation_labels = validation_data["sensitivity"]
test_labels = testing_data["sensitivity"]


nlp = xx_ent_wiki_sm.load(disable=['ner'])

vectorizer = CountVectorizer(tokenizer=tokenize_normalize, binary=True)
trainD = vectorizer.fit_transform(training_data["words"])

gModel = GaussianNB(var_smoothing=0.000000000000000001)
gModel.fit(trainD.toarray(), train_labels)

fetchSQL = 'SELECT word, id FROM WordNetEval6'
cursor.execute(fetchSQL)

data = cursor.fetchall()

dataFrame = pd.DataFrame(data, columns=("words", "id"))

predictSensitivity(dataFrame, vectorizer, gModel, cursor)
db.commit()