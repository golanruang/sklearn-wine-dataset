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

# draw a couple of graphs of variables vs. output variable
# look at how the graphs change before/after you scale it
# look at some variables (a reasonable amount)

# why is the standard scaler so effective? 81% --> 100%
# look at incoming data and see why

# look at GaussianNB - conditional probability
# probability of one thing changes based on the outcome of another thing

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
