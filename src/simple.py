from codt_py import OptimalDecisionTreeClassifier
from juliacall import Main as jl
from pycontree import ConTree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from .util import read_quant_dataset

def main():

    Xtrain, Ytrain, Xtest, Ytest = read_quant_dataset("bank")
    X, y = (np.concatenate([Xtrain, Xtest]), np.concatenate([Ytrain, Ytest]))

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
    misses, tree = jl.optimal_classification_2d(X, y_quant)
    print("Quant-BnB Accuracy: ", 1.0 - (misses / len(y)))
