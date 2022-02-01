import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import Colormap, ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from hw5code import DecisionTree, find_best_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data')
for column_name in df.columns:
    column = df[column_name]
    df[column_name] = LabelEncoder().fit_transform(column)

X, y = df.drop(columns=['p']).values, df['p'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
clf = DecisionTree(feature_types=['categorical' for _ in range(X.shape[1])])
clf.fit(X_train, y_train)
y_pred_test = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('test accuracy: {}'.format(test_accuracy))
