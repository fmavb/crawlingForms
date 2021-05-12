
from Levenshtein import distance
import os
import pandas as pd
import numpy as np
import mysql.connector

class LevenshteinKNN:
    train_features = []
    train_labels = []
    neighbours = 5
    
    def __init__(self, neighbours=5):
        self.neighbours = neighbours
    
    def fit(self, features, labels):
        self.train_features = features
        self.train_labels = labels
        
    def predict(self, features):
        predictions = []
        for item in features:
            distances = []
            for i in range(len(self.train_features)):
                distances.append([i, distance(item, self.train_features[i])])
            sortDist = sorted(distances, key=lambda x:x[1])
            votes = {}
            for i in range(self.neighbours):
                if votes.get(self.train_labels[sortDist[i][0]], None):
                    votes[self.train_labels[sortDist[i][0]]] += 1
                else:
                    votes[self.train_labels[sortDist[i][0]]] = 1
            
            resultPred = None
            votesMax = None
            for key, value in votes.items():
                if resultPred is None and votesMax is None:
                    resultPred = key
                    votesMax = value
                    continue
                if value > votesMax:
                    resultPred = key
                    votesMax = value
                    
            predictions.append(resultPred)
            
        return predictions

def predictSensitivity(fields, model, cursor):
    sqlKNN = "UPDATE WordNetEval7 SET KNN=%s WHERE id=%s"
    features = []
    for item in fields["words"]:
        features.append(item)

    resultsLabels = model.predict(features)
    
    toDBKNN = []

    for i in range(len(resultsLabels)):
        toDBKNN.append((resultsLabels[i], fields['id'][i].item()))
        
        print("Row %s out of %s rows" % (i, len(resultsLabels)))

    cursor.executemany(sqlKNN, toDBKNN)
    

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


model = LevenshteinKNN(neighbours=2)

trainData = []

for item in training_data["words"]:
    trainData.append(item)
    
trainLabels = []
for item in train_labels:
    trainLabels.append(item)


model.fit(trainData, trainLabels)


fetchSQL = 'SELECT word, id FROM WordNetEval7'
cursor.execute(fetchSQL)

data = cursor.fetchall()

dataFrame = pd.DataFrame(data, columns=("words", "id"))

predictSensitivity(dataFrame, model, cursor)
db.commit()