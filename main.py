import sklearn
from sklearn.utils import shuffle
from sklearn import datasets
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Load libraries
import numpy
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Data file import
playlist_data = pd.read_csv("playlist.csv")

# Attribute to be predicted
predict = "TP"
# Dataset/Column to be Predicted, X is all attributes and y is the features
# x = np.array(playlist_data.drop([predict], 1)) # Will return a new data frame that doesnt have  in it
# y = np.array(playlist_data[predict])

le = preprocessing.LabelEncoder()

album_release_date = le.fit_transform(list(playlist_data["album_release_date"]))  # album_release_date
number_of_tracks_in_album = le.fit_transform(list(playlist_data["number_of_tracks_in_album"]))  # number_of_tracks_in_album
position_in_playlist = le.fit_transform(list(playlist_data["position_in_playlist"]))  # position_in_playlist
track_duration_ms = le.fit_transform(list(playlist_data["track_duration_ms"]))  # track_duration_ms
track_popularity = le.fit_transform(list(playlist_data["track_popularity"]))  # track_popularity
track_explicit = le.fit_transform(list(playlist_data["track_explicit"]))  # track_explicit
images_path = le.fit_transform(list(playlist_data["images_path"]))  # images_path
data_collection = le.fit_transform(list(playlist_data["data_collection"]))  # data_collection
x = list(zip(album_release_date, number_of_tracks_in_album, position_in_playlist, track_duration_ms, track_popularity,track_explicit, images_path, data_collection))
y = list(data_collection)
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = 'accuracy'

# The test data will test the accuracy of the model created
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.20, random_state=seed)

# Check with  different Scikit-learn classification algorithms
models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    msg += '\n'
    print(msg)

# Compare Algorithms' Performance
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#ensembles
ensembles=[]
ensembles.append(('ScaledAB',Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostRegressor())])))
ensembles.append(('ScaledGBM',Pipeline([('Scaler',StandardScaler()),('GBM',GradientBoostingRegressor())])))
ensembles.append(('ScaledRF',Pipeline([('Scaler',StandardScaler()),('RF',RandomForestRegressor())])))
ensembles.append(('ScaledET',Pipeline([('Scaler',StandardScaler()),('ET',ExtraTreesRegressor())])))
results=[]
names=[]
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#Comparing Scaled Ensemble ALgorithms
fig= plt.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparsion')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#Model Evaluation Metric 4-prediction report
for x in range(len(predictions)):
  print("Predicted: ",predictions[x],'\t\t' "Actual: ", Y_validation[x],)
