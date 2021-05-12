from os import stat
import mysql.connector

def stats(model, fieldName1, fieldName2):
    svmSQL = 'SELECT DISTINCT ff.token, word, type, predictedSensitivity, %s FROM WordNetEval7 as wn, form_fields2 as ff WHERE wn.token = ff.token AND ff.model = "%s"' % (fieldName1, fieldName2)

    cursor.execute(svmSQL)
    testSVM = cursor.fetchall()

    numHyper = 0
    numHypo = 0

    correctHyper = 0
    correctHypo = 0
    incorrectHyper = 0
    incorrectHypo = 0
    classHyperDict = {
        1:{
            1:0,
            2:0,
            3:0
            },
        2:{
            1:0,
            2:0,
            3:0
        },
        3:{
            1:0,
            2:0,
            3:0
        }
    }
    classHypoDict = {
        1:{
            1:0,
            2:0,
            3:0
        },
        2:{
            1:0,
            2:0,
            3:0
        },
        3:{
            1:0,
            2:0,
            3:0
        }
    }
    for item in testSVM:
        if item[2] == "hypernym":
            classHyperDict[item[3]][item[4]] += 1
            
            numHyper += 1
            
            if item[3] <= item[4]:
                correctHyper += 1
            else:
                incorrectHyper += 1

        elif item[2] == "hyponym":
            classHypoDict[item[3]][item[4]] += 1
            numHypo += 1

            if item[3] >= item[4]:
                correctHypo +=1
            else:
                incorrectHypo += 1

    print(model)
    print(classHyperDict)
    print("\n")
    print(classHypoDict)
    print("\n")
    print("Number of hypernyms: " + str(numHyper))
    print("Number of hyponyms: " + str(numHypo))
    print("Correct hypernym classification: " + str(correctHyper))
    print("Incorrect hypernym classification: " + str(incorrectHyper))
    print("Correct hypernym ratio: " + str(correctHyper/numHyper))
    print("Correct hyponym classification: " + str(correctHypo))
    print("Incorrect hyponym classification: " + str(incorrectHypo))
    print("Correct hyponym ratio: " + str(correctHypo/numHypo))
    print("------------------------------")


db = mysql.connector.connect(host="localhost",
user="root", password="yourpassword", database="ontology")

cursor = db.cursor()

cursor.execute("set @@sql_mode='NO_ENGINE_SUBSTITUTION';")
db.commit()

stats("Support Vector Machine", "SVM", "Support Vector Machine")
stats("Decision Trees", "Tree", "Decision Trees")
stats("Logistic Regression", "Logistic", "Logistic Regression")
stats("Naive Bayes", "Bayes", "Naive Bayes")
stats("KNN", "KNN", "KNN")