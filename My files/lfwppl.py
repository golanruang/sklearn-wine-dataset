from time import time
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import svm

#print(__doc__)

lfw_people=fetch_lfw_people(min_faces_per_person=70,resize=0.4)
data=lfw_people.data
target=lfw_people.target

trainX, testX, trainY, testY = train_test_split(data,target,test_size=0.3, random_state=42)
clf=svm.SVC()
clf.fit(trainX,trainY)
pred_test = clf.predict(testX)
correct = 0
incorrect = 0
for pred, gt in zip(pred_test, testY):
    if pred == gt: correct += 1
    else: incorrect += 1
print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")
