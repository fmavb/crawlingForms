import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import mysql.connector
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import xx_ent_wiki_sm
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

def fileParser(fileName, dir, vectorizer, model, cursor, fileCounter, db):
    openFile = open(fileName)
    try:
        text = openFile.read()

        fileNameSQL = 'INSERT INTO forms2 (id, fileName, website) VALUES (%s, %s, %s)'
        tokenSQL = 'INSERT INTO form_fields2 (token, fieldType, predictedSensitivity, formID, model) VALUES (%s, %s, %s, %s, %s)'

        #cursor.execute(fileNameSQL,(fileCounter, fileName, dir))
        html = BeautifulSoup(text, features="lxml")
        labels = html.find_all("label")

        listLabels = []

        for label in labels:
            id = label.get("for")
            if id:
                labelToInput = html.find(id=id)
                if labelToInput and labelToInput.get("type") == "hidden":
                    listLabels.append([label.decode_contents(), 0, "label hidden inputID: %s" % (id)])
                    continue
                if labelToInput and labelToInput.get("type") != "hidden":
                    listLabels.append([label.decode_contents(), 0, "label inputID: %s" % (id)])
                    continue
            content = label.decode_contents()
            if not isinstance(content, str):
                raise Exception("Content is not a string %s" % (content))
            listLabels.append([content, 0])
        
        dataFrameLabels = pd.DataFrame(data=listLabels, columns=("words", "sensitivity", "type"))
        
        if len(listLabels) > 0:
            predictD = vectorizer.transform(dataFrameLabels["words"])
            resultsLabels = model.predict(predictD.toarray())

            labelsToDB = []
            for i in range(len(resultsLabels)):
                labelsToDB.append((dataFrameLabels["words"][i], dataFrameLabels["type"][i], int(resultsLabels[i]), fileCounter, "Naive Bayes"))

            cursor.executemany(tokenSQL, labelsToDB)

        inputs = html.find_all("input")
        listInputs = []

        for input in inputs:
            if input.get("type") == "hidden":
                placeholder = input.get("placeholder")
                if placeholder:
                    listInputs.append([placeholder, 0, "input placeholder hidden"])
            else:
                placeholder = input.get("placeholder")
                if placeholder:
                    listInputs.append([placeholder, 0, "input placeholder"])

        
        dataFrameInputs = pd.DataFrame(data=listInputs, columns=("words", "sensitivity", "type"))
        if len(listInputs) > 0:
            predictD = vectorizer.transform(dataFrameInputs["words"])
            resultsInputs = vectorizer.predict(predictD.toarray())

            inputsToDB = []
            for i in range(len(dataFrameInputs)):
                inputsToDB.append((dataFrameInputs["words"][i], dataFrameInputs["type"][i], int(resultsInputs[i]), fileCounter, "Naive Bayes"))


            cursor.executemany(tokenSQL, inputsToDB)
        db.commit()
    except UnicodeDecodeError:
        print("Unicode error")
    finally:
        openFile.close()
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

objects = os.listdir(os.getcwd())
dirs = []
for object in objects:
    if os.path.isdir(os.path.join(os.getcwd(), object)):
        dirs.append(object)

dirs.remove("spiders")

fileCounter = 0
for dir in dirs:
    allFiles = os.listdir(os.path.join(os.getcwd(), dir))
    for i in range(len(allFiles)):
        fileParser(os.path.join(os.getcwd(), dir, allFiles[i]), dir, vectorizer, gModel, cursor, fileCounter, db)
        fileCounter += 1
        print("%s files completed out %s files" % (i, len(allFiles)))
