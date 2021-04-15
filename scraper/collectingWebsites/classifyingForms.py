import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import mysql.connector
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import xx_ent_wiki_sm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression



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

def fileParser(fileName, dir, modelSVM, modelTree, modelLogis, cursor, fileCounter, db):
    openFile = open(fileName)
    try:
        text = openFile.read()

        fileNameSQL = 'INSERT INTO forms2 (id, fileName, website) VALUES (%s, %s, %s)'
        tokenSQL = 'INSERT INTO form_fields2 (token, fieldType, predictedSensitivity, formID, model) VALUES (%s, %s, %s, %s, %s)'

        cursor.execute(fileNameSQL,(fileCounter, fileName, dir))
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
            listLabels.append([content, 0, "label"])
        
        dataFrameLabels = pd.DataFrame(data=listLabels, columns=("words", "sensitivity", "type"))
        
        if len(listLabels) > 0:
            resultsLabels = modelSVM.predict(dataFrameLabels)
            resultsLabels1 = modelTree.predict(dataFrameLabels)
            resultsLabels2 = modelLogis.predict(dataFrameLabels)

            labelsToDB = []
            for i in range(len(resultsLabels)):
                labelsToDB.append((dataFrameLabels["words"][i], dataFrameLabels["type"][i], int(resultsLabels1[i]), fileCounter, "Support Vector Machine"))
                labelsToDB.append((dataFrameLabels["words"][i], dataFrameLabels["type"][i], int(resultsLabels1[i]), fileCounter, "Decision Trees"))
                labelsToDB.append((dataFrameLabels["words"][i], dataFrameLabels["type"][i], int(resultsLabels2[i]), fileCounter, "Logistic Regression"))

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

        
        dataFrameInputs = pd.DataFrame(data=listInputs, columns=("words", "sensitivity"))
        if len(listInputs) > 0:
            resultsInputs = modelSVM.predict(dataFrameInputs)
            resultsInputs1 = modelTree.predict(dataFrameInputs)
            resultsInputs2 = modelLogis.predict(dataFrameInputs)

            inputsToDB = []
            for i in range(len(dataFrameInputs)):
                inputsToDB.append((dataFrameInputs["words"][i], dataFrameInputs["type"][i], int(resultsInputs[i]), fileCounter, "Support Vector Machine"))
                inputsToDB.append((dataFrameInputs["words"][i], dataFrameInputs["type"][i], int(resultsInputs1[i]), fileCounter, "Decision Trees"))
                inputsToDB.append((dataFrameInputs["words"][i], dataFrameInputs["type"][i], int(resultsInputs2[i]), fileCounter, "Logistic Regression"))


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


pipeSVM = Pipeline([("word", Pipeline([('selector', ItemSelector(key='words')), 
                                         ('tfidf', TfidfVectorizer(tokenizer=tokenize_normalize))])),
                      ("SVM", SVC(kernel="linear", break_ties=True, max_iter=-1, C=100, tol=0.1))])

pipeSVM.fit(training_data, train_labels)

pipeTree = Pipeline([("word", Pipeline([('selector', ItemSelector(key='words')), 
                                         ('tfidf', CountVectorizer(tokenizer=tokenize_normalize,binary=True))])), 
                      ("tree", DecisionTreeClassifier())])

pipeTree.fit(training_data, train_labels)

pipeLogis = Pipeline([("word", Pipeline([('selector', ItemSelector(key='words')), 
                                         ('tfidf', CountVectorizer(tokenizer=tokenize_normalize,binary=True))])), 
                      ("lr", LogisticRegression(C=10, max_iter=200, penalty='l2',solver='newton-cg'))])

pipeLogis.fit(training_data, train_labels)

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
        fileParser(os.path.join(os.getcwd(), dir, allFiles[i]), dir, pipeSVM, pipeTree, pipeLogis, cursor, fileCounter, db)
        fileCounter += 1
        print("%s files completed out %s files" % (i, len(allFiles)))
