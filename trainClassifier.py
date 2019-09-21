#%%
import pandas as pd
#from treelib import Node, Tree
#from categoryTree import cu

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.metrics import classification_report
import pickle as pkl

#%%

#Inputs: df: dataFrame from csv, lvl: level of tree to be displayed
#DIsplays the population distribution of the labeled items in a given level 
def show_level_stats(df, level):
    lvl = 'level ' + str(level)
    df= pd.DataFrame(df)
    print('level specific: ')
    print((df.groupby(lvl).size()/df.groupby(lvl).size().sum())*100)
    print('over all population: ')
    print((df.groupby(lvl).size()/len(df))*100)
    df.groupby(lvl).size().plot(kind = 'bar', figsize =(15, 8))
    plt.show()

#inputs:clf: trained classified, x_test: target features, y_test: target y values, labels: labels in the dataset
#displays the metrics of the performance of the classifier
def show_metrics(clf, x_test, y_test, labels):
    print('Accuracy Score: {}'.format(clf.score(x_test, y_test)))
    print('Confusion Matrix: ')
    pred = clf.predict(x_test)
    
    cm1 = cm(y_test, pred, labels)
    cm1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
    print(cm1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm1,cmap='Blues')
    
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print('Metric Overview: ')
    print(classification_report(y_test, pred))

#inputs: df: dataframe from csv, lvl: level in tree to be trained, showMetrics: whether to show performance metrics of the classifier
#   saveName: file name to save classifier after training, if no name is passed, classifier is not saved
#   params: hyperparameters for the classifier, if no parameters are passed, default params are used
#   showLevelStats: show population statistics of the level to be trained
#   fillna: replaces na values with placeholder, if no value is passed, all na values are dropped
#FUnction trains and returns a random forest classifier
def train_rf(df, lvl, showMetrics=False, saveName=None, params = {}, showLevelStats = False, fillna=None):
    level = 'level ' + str(lvl)
    clf = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('clf', rf().set_params(**params)),
                ])
    if(fillna is None):
        df = df[[df.columns[0], level]].dropna()
    else:
        df = df.fillna(fillna)
    x=df[df.columns[0]].values
    if(showLevelStats==True):
        show_level_stats(df, lvl)
    y=df[level].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1000)
    clf.fit(x_train, y_train)
    if(showMetrics ==True):
        show_metrics(clf, x_test, y_test, labels = list(df[level].unique()))
    if(saveName is None):
        return(clf)
    else:
        try:
            print('saving.......')
            file = saveName +'.pkl'
            pkl.dump(clf, open(file, 'wb'))
        except Exception as e:
            print(e)

#inputs: df: dataframe from csv, levels: list of the levels in the tree to be trained, showMetrics: whether to show performance metrics of the classifier
#   saveName: file name to save classifier after training, if no name is passed, classifier is not saved
#   params: hyperparameters for the classifier, if no parameters are passed, default params are used
#   showLevelStats: show population statistics of the level to be trained
#   fillna: replaces na values with placeholder, if no value is passed, all na values are dropped
#FUnction trains and returns an array of random forest classifiers. The order of the classifiers in the array is the order of the levels in the levels array input to the function
def train_levels(df, levels, showMetrics=False, saveName=None, params = {}, showLevelStats = False, fillna=None):

    clfs = []
    for j in levels:
            try:
                clf = train_rf(df, j, showMetrics, saveName, params, showLevelStats, fillna)
                clfs.append(clf)
            except Exception as e:
                print('Could not train level {}'.format(j))
                print(e)
                
    return(clfs)

#%%
#############################################
#Example code below
#loading a dataset
#mi_df = pd.read_csv('Main Ingredient(refined).csv')

#%%
#training a single classifier for a specific level
#clf = train_rf(mi_df, 3, showMetrics=True, saveName='milvl3', params = {}, showLevelStats = True, fillna=None)

#%%
#inspecting the population of a specific level in dataset
#show_level_stats(mi_df, 3)

#%%
#training multiple classifiers
#clfs = train_levels(df, [1, 2, 3], showMetrics = True, showLevelStats = False)


#%%
