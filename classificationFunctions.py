#%%

import string
from categoryTree import cu
from treelib import Tree,Node
import pickle as pkl
import numpy as np

#%%


#Input#Inputs: tree: the built classification tree, pred: list of predictions in the form [pred1, pred2...predn]
#Returns list of preds, with those preds removed that are the descendant of any other pred

def trimResults(preds, tree):
    preds = list(set(preds))
    temp = preds.copy()
    
    for i in preds:
    #temp = result.copy()

        for j in preds:
            if (tree.is_ancestor(j, i)  ):
                try:
                    temp.remove(j)
  
                except:
                    pass
            elif (tree.is_ancestor(i, j)):
                try:
                    temp.remove(i)
                
        
                except:
                    pass
    return(temp)

#Input: tree: the built classification tree, pred: list of predictions in the form [pred1, pred2...predn]
# returns the results as a list of lists, doing a depth first search till the result class is reached in the form:
#[[result 1 level 1,result 1 level 2], [result 2 level 1,result 2 level 2]]

def resolveResults(preds, tree):
    cats = []
    for i in preds:
        cats.append([tree.get_node(x).identifier for x in tree.rsearch(i)][::-1])

    return cats
# Input: clf: list of classifiers in order of the levels eg:[clflvl1, clflvl2....]
#     tree: the built classification tree
#     value: value to be labelled
#     confidenceLimit: the limit below which the system will reject a prediction from the classifier
#     Note that the first classifier in the list (clf) is not subject to this limit
# Returns a fully resolved result eg: ['resul 1', 'descendant 1', 'descendant 2', 'decendant 3']
def getClassifierPreds(clfs, tree, value, confidenceLimit = 0.4):
    classes = []
    for i in clfs:
        classes.append(i.classes_)
    

    preds = []
 
    for i in clfs:
        preds.append(i.predict_proba([value])[0])

    #print(preds)

    sortedArgPreds = [np.argsort(i)[::-1] for i in preds]

    finalPreds = classes[0][sortedArgPreds[0][0]]


    for i,j in enumerate(clfs[1:]):

        for k, l in enumerate(sortedArgPreds[i+1]):
            b = classes[i+1][sortedArgPreds[i+1][k]]
            if tree.is_ancestor(finalPreds.lower(), b.lower()) and preds[i+1][l]>=confidenceLimit:
                finalPreds=b
                break

    return resolveResults([finalPreds.lower()], tree)

#Input: model name (.pkl is appended automatically)
#Returns loaded classifier
def loadModel(name):
    try:
        with open(name+'.pkl', 'rb') as fid:
            clf1 = pkl.load(fid)
        return clf1
    except Exception as e:
        print(e)
        return 0

#Input: Results: List of results in the form: [[[classifier result 1 level 1,classifier result 1 level 2]], [[name search result 1 level 1,result 1 level 2], [name searcg result 2 level 1,result 2 level 2]]]
#     Priority: either 1 or 0, if 1, Priority is given to the results from the second source in the results list, and vice versa
#     tree: the built classification tree
# returns the combined result from both sources 

        
#%%
########################################################


#%%
