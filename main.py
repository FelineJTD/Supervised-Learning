# Imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import sys
sys.path.insert(0, './src')

from script_1 import KNN
from script_2 import logistic_regression
from script_3 import ID3

import argparse

parser = argparse.ArgumentParser(description='Program implementing the KNN, Logistic Regression, and ID3 algorithms.')
parser.add_argument('--model', default='knn', choices=['knn', 'log_reg', 'id_3'],
                    help='Select the model to use.')
parser.add_argument('--data', default='iris', choices=['iris', 'digits'],
                    help='Select the dataset to use.')
parser.add_argument('--k_nearest', default=3, type=int, help='Specify the number of neighbors to use.')
parser.add_argument('--lr', default=0.3, type=float, help='Specify the learning rate to use.')
parser.add_argument('--epochs', default=1000, type=int, help='Specify the number of epochs to use.')

args = parser.parse_args()

method = args.model

def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

data = datasets.load_iris()
X, y = data.data, data.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1234
)

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)

# Plot the training data, uncomment to see it
# from matplotlib.colors import ListedColormap
# import matplotlib.pyplot as plt
# cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
# plt.figure(figsize=(10, 10))
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap)
# plt.show()

if method == 'knn':
  k = 3
  model = KNN(k=k)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  print("KNN classification accuracy", accuracy(y_test, predictions))

elif method == 'log_reg':
  lr = args.lr
  epochs = args.epochs
  model = logistic_regression(lr=lr, epochs=epochs)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  print("Logistic regression classification accuracy", accuracy(y_test, predictions))

elif method == 'id_3':
  model = ID3()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  print("ID3 classification accuracy", accuracy(y_test, predictions))

else:
  print("Invalid method")
  exit(1)