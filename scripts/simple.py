from codt_py import OptimalDecisionTreeClassifier
from juliacall import Main as jl
from pycontree import ConTree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

df = pd.read_csv("./contree/datasets/bank.txt", sep=" ", header=None)

X = df[df.columns[1:]]
y = df[0]

max_depth=2

codt = OptimalDecisionTreeClassifier(max_depth=max_depth)
contree = ConTree(max_depth=max_depth, verbose=True)
cart = DecisionTreeClassifier(max_depth=max_depth)

codt.fit(X, y)
contree.fit(X, y)
cart.fit(X, y)

codt_ypred = codt.predict(X)
print("CODT Accuracy: " , accuracy_score(y, codt_ypred))

contree_ypred = contree.predict(X)
print("ConTree Accuracy: " , accuracy_score(y, contree_ypred))

cart_ypred = cart.predict(X)
print("CART Accuracy: " , accuracy_score(y, cart_ypred))

jl.seval("include(\"Quant-BnB/call.jl\")")
y_quant = np.zeros((y.size, y.max() + 1))
y_quant[np.arange(y.size), y] = 1
print("Quant-BnB Accuracy: ", jl.optimal_classification_2d(X.to_numpy(), y_quant))
