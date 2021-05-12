import os
from bs4 import BeautifulSoup
import pandas as pd
import mysql.connector
from Levenshtein import distance


def fileParser(fileName, dir, model, cursor, fileCounter, db):
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
            
            features = []
            for item in dataFrameLabels["words"]:
                features.append(item)

            resultsLabels = model.predict(features)

            labelsToDB = []
            for i in range(len(resultsLabels)):
                labelsToDB.append((features[i], dataFrameLabels["type"][i], int(resultsLabels[i]), fileCounter, "KNN"))

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

            features = []
            for item in dataFrameLabels["words"]:
                features.append(item)

            resultsLabels = model.predict(features)

            inputsToDB = []
            for i in range(len(dataFrameInputs)):
                inputsToDB.append((features[i], dataFrameInputs["type"][i], int(resultsLabels[i]), fileCounter, "KNN"))


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

model = LevenshteinKNN(neighbours=2)

trainData = []

for item in training_data["words"]:
    trainData.append(item)
    
trainLabels = []
for item in train_labels:
    trainLabels.append(item)


model.fit(trainData, trainLabels)

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
        fileParser(os.path.join(os.getcwd(), dir, allFiles[i]), dir, model,cursor, fileCounter, db)
        fileCounter += 1
        print("%s files completed out %s files" % (i, len(allFiles)))