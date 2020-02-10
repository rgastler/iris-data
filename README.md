The following represents my learning process while considering the famous iris dataset.  The dataset consists of 150 samples of various measurements of iris flowers and their species (setosa, versicolor, virginica).  The measurements are sepal width, sepal length, petal width and petal length.  The problem is to create a classification model that will predict the species given the four feature measurements.  The file 'iris data analysis.ipynb' was created using Google's Colaboratory, which is very similar to the Jupyter Notebook.  The file should run easily in either and the iris dataset comes included with scikitlearn.

The following dependencies are included.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

In the analysis, we load the dataset into a pandas dataframe, explore the data with a few plots using matplotlib, and then train, test, and compare three different classification models: K Nearest Neighbors, Support Vector Machine, and Random Forest Classifier.  Finally, just for fun, I create a couple test points to compare how the model predicts them and compared the results with our guess by just 'eye balling' the plots.

Robert R. Gastler
