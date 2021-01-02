# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 02:10:31 2020

@author: abdo
"""
# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv('zoo.csv')
data=dataset.head()
X=dataset.iloc[:,1:17].values
y=dataset.iloc[:,17].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)

# Predicting the Test set results
classifier.predict(X_test[15:25])
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0][0] + cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4] + cm[5][5]+cm[6][6]) / (cm.sum().sum())

# Visualising
sb.distplot(dataset['class_type'],kde = False)
plt.show()
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(14)]).T
# # Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
# pred = classifier.predict(Xpred).reshape(X1.shape)
# plt.contourf(X1, X2, pred,
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green','blue','yellow','black'))(i), label = j)
# plt.title('SVM (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
