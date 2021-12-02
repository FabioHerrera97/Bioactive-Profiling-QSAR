# Importing the required libraries

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_roc_curve

# Uploading the data_cleaned file

file = 'Path where the file with the cleaned data is'
data = pd.read_excel(file, sheet_name='Sheet1')
data

# Drop the columns corresponding to the name and the SMILES

to_drop =['Name','SMILES']
data = data[data.columns.difference(to_drop)]
data.head()

# Splitting the training and test datasets 

columns_names = data.columns.values
X = data.drop(['Bioactive'], axis = 1)
y = data['Bioactive']
columns_names = X.columns.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

#Training a Support Vector Machine (svm)

from sklearn.svm import LinearSVC
svm=LinearSVC(C=0.0001)
svm.fit(X_train, y_train)

# Performance of SVM

print("score on test: " + str(svm.score(X_test, y_test)))
print("score on train: "+ str(svm.score(X_train, y_train)))
y_pred_svm = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_svm)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_test, y_pred_svm)
print('Precision: %f' % precision)
recall = recall_score(y_test, y_pred_svm)
print('Recall: %f' % recall)
f1 = f1_score(y_test, y_pred_svm)
print('F1 score: %f' % f1)
auc = roc_auc_score(y_test, y_pred_svm)
print('ROC AUC: %f' % auc)
rdf_disp = plot_roc_curve(svm, X_test, y_test)
plt.show()

# Training a Random Forest (RF)

from sklearn.ensemble import RandomForestClassifier
# n_estimators = number of decision trees
rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

# Performance of RF

y_pred_rf = rf.predict(X_test)
print("score on test: " + str(rf.score(X_test, y_test)))
print("score on train: "+ str(rf.score(X_train, y_train)))
accuracy = accuracy_score(y_test, y_pred_rf)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_test, y_pred_rf)
print('Precision: %f' % precision)
recall = recall_score(y_test, y_pred_rf)
print('Recall: %f' % recall)
f1 = f1_score(y_test, y_pred_rf)
print('F1 score: %f' % f1)
auc = roc_auc_score(y_test, y_pred_rf)
print('ROC AUC: %f' % auc)
rdf_disp = plot_roc_curve(rf, X_test, y_test)
plt.show()

# Training a Logistic Regression (LR)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Performance of LR

print("score on test: " + str(lr.score(X_test, y_test)))
print("score on train: "+ str(lr.score(X_train, y_train)))
y_pred_lr = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_lr)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_test, y_pred_lr)
print('Precision: %f' % precision)
recall = recall_score(y_test, y_pred_lr)
print('Recall: %f' % recall)
f1 = f1_score(y_test, y_pred_lr)
print('F1 score: %f' % f1)
auc = roc_auc_score(y_test, y_pred_lr)
print('ROC AUC: %f' % auc)
rdf_disp = plot_roc_curve(lr, X_test, y_test)
plt.show()

# Training a Decision Tree (DT)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Performance of DT

print("score on test: "  + str(clf.score(X_test, y_test)))
print("score on train: " + str(clf.score(X_train, y_train)))
y_pred_clf = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_clf)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_test, y_pred_clf)
print('Precision: %f' % precision)
recall = recall_score(y_test, y_pred_clf)
print('Recall: %f' % recall)
f1 = f1_score(y_test, y_pred_clf)
print('F1 score: %f' % f1)
auc = roc_auc_score(y_test, y_pred_clf)
print('ROC AUC: %f' % auc)
rdf_disp = plot_roc_curve(clf, X_test, y_test)
plt.show()
