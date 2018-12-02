# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 01:19:32 2018

@author: HP
"""

import logging
import random

from DecisionTree import DecisionTreeClassifier

class RandomForestClassifier(object):
    
    """
    parameter  rf_trees:    Number of decision trees to use
    parameter  rf_samples:  Number of samples to give to each tree
    parameter  max_depth:   Maximum depth of the trees
    """
    def __init__(self, rf_trees, rf_samples, max_depth=-1):
        self.trees = []
        self.rf_trees = rf_trees
        self.rf_samples = rf_samples
        self.max_depth = max_depth
    
    """
    fit : Trains self.rf_trees number of decision trees.
    :parameter   data: A list of lists with the last elemen the value to predict
    """
    def fit(self, data):
        #sample_fts = list( map(lambda x: [x, random.sample(data, self.rf_samples)], range(self.rf_trees)))
        #print (sample_fts)
        sample_fts = [random.sample(data,self.rf_samples) for x in range(self.rf_trees)]
        self.trees= list(map(self.train_tree, sample_fts))
    """
    train_tree : Trains a single tree and returns it
    :parameter   data: A list containing the index of the tree being trained and
                       the data to train it
    """
    
    def train_tree(self, data):
        tree = DecisionTreeClassifier(max_depth = self.max_depth)
        print(1)
        tree.fit(data)
        return tree
    """
    predict : Returns a prediction for the given feature. The result is the value
    that gets the most votes.
    :parameter   feature: the features used to predict
    """
    def predict(self, feature):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(feature))
        return max(set(predictions), key =predictions.count)