import mysql.connector
import os
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import concurrent

def hypernyms(field, cursor):
    wordArray = field.split(" ")
    stopWordsRemoved = [word for word in wordArray if word not in stopwords.words('english')]

    toDB = []
    sql = "INSERT INTO WordNetEval6 (token, exactWord, word, type) VALUES (%s, %s, %s, %s)"
    
    synsets = wn.synsets(field.replace(" ", "_"))
    if len(synsets) > 0:
        for synset in synsets:
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    toDB.append((field, field, str(lemma.name()), "hypernym"))
    else:
        for word in stopWordsRemoved:
            for synset in wn.synsets(word):
                for hypernym in synset.hypernyms():
                    for lemma in hypernym.lemmas():
                        toDB.append((field, word, str(lemma.name()), "hypernym"))
    
    cursor.executemany(sql, toDB)
    return

def hyponyms(field, cursor):
    
    wordArray = field.split(" ")
    stopWordsRemoved = [word for word in wordArray if word not in stopwords.words('english')]
    
    toDB = []
    sql = "INSERT INTO WordNetEval6 (token,exactWord, word, type) VALUES (%s,%s, %s, %s)"
    #for word in stopWordsRemoved:
    synsets = wn.synsets(field.replace(" ", "_"))
    if len(synsets) > 0:
        for synset in synsets:
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    toDB.append((field, field, str(lemma.name()), "hyponym"))
    else:
        for word in stopWordsRemoved:
            for synset in wn.synsets(word):
                for hyponym in synset.hyponyms():
                    for lemma in hyponym.lemmas():
                        toDB.append((field, word, str(lemma.name()), "hyponym"))
    cursor.executemany(sql, toDB)
    return
    

db = mysql.connector.connect(host="localhost",
user="root", password="yourpassword", database="ontology")

cursor = db.cursor()

cursor.execute("set @@sql_mode='NO_ENGINE_SUBSTITUTION';")
db.commit()

fetchSQL = "SELECT DISTINCT token FROM form_fields"

cursor.execute(fetchSQL)

formFields = cursor.fetchall()

rowsNumber = len(formFields)

for i in range(len(formFields)):
    hypernyms(formFields[i][0], cursor)
    hyponyms(formFields[i][0], cursor)
    db.commit()
    print("%s rows completed out %s rows" % (i, rowsNumber))
