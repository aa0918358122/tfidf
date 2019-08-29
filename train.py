# load x.npy, y.npy
# select 6 article as test data
# fit model, random forest sklearn
# evaluate test data

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

x = np.load('x.npy')
y = np.load('y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = None, shuffle = True, stratify = y)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

clf = RandomForestClassifier(n_estimators = 20, random_state = 0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
