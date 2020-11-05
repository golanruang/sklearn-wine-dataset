# python 3.8
# Scikit-learn ver. 0.23.2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix

from sklearn.datasets import load_wine

from matplotlib import pyplot

#print(matplotlib.__version__)

features, target = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=42)

unscaledClf=make_pipeline(PCA(n_components=2),GaussianNB())
unscaledClf.fit(X_train, y_train)
pred_test=unscaledClf.predict(X_test)

correct =0
incorrect=0
for pred,gt in zip(pred_test,y_test):
    if pred==gt:
        correct+=1
    else:
        incorrect+=1
print(f"\nUnscaled Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}\n")
plot_confusion_matrix(unscaledClf,X_test,y_test)

scaledClf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
scaledClf.fit(X_train, y_train)
pred_test = scaledClf.predict(X_test)

correct =0
incorrect=0
for pred,gt in zip(pred_test,y_test):
    if pred==gt:
        correct+=1
    else:
        incorrect+=1
print(f"PCA Scaled Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")
plot_confusion_matrix(unscaledClf,X_test,y_test)

scaledClf=make_pipeline(StandardScaler(),GaussianNB(priors=None))
scaledClf.fit(X_train, y_train)
pred_test=scaledClf.predict(X_test)
correct =0
incorrect=0
for pred,gt in zip(pred_test,y_test):
    if pred==gt:
        correct+=1
    else:
        incorrect+=1
print(f"\nStandard Scaled Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}\n")
plot_confusion_matrix(unscaledClf,X_test,y_test)
pyplot.show()
#array([0, 1, 2])

#['class_0', 'class_1', 'class_2']
# wine = load_wine()
# print(wine.head)

# wineX = wine.data
# wineY = wine.target
# clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=10000, tol=1e-3))
# clf.fit(wineX,wineY)
#
# print(clf.predict([[-0.8, -1]]))

# trainX, testX, trainY, testY = train_test_split(
#     digitsX, digitsY, test_size = 0.3, shuffle = True
#     )
#
# classifier = LogisticRegression(max_iter = 10000)
# classifier.fit(trainX, trainY)
# preds = classifier.predict(testX)
#
# correct = 0
# incorrect = 0
# for pred, gt in zip(preds, testY):
#     if pred == gt: correct += 1
#     else: incorrect += 1
# print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")
#
# plot_confusion_matrix(classifier, testX, testY)
# pyplot.show()
