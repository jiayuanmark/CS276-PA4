#!/usr/bin/env python

### Module imports ###
import sys
import math
import re
import numpy as np
from sklearn import linear_model, svm, preprocessing
from utilFunction import *
import prank

###############################
##### Point-wise approach #####
###############################
def pointwise_train_features(train_data_file, train_rel_file):
  queries, documents = read_feature_file(train_data_file)
  X, qryDocList = build_features(queries, documents)
  y = build_labels(train_rel_file, qryDocList)
  return (X, y)
 
def pointwise_test_features(test_data_file):
  queries, documents = read_feature_file(test_data_file)
  X, qryDocList = build_features(queries, documents)
  queries, index_map = build_indexmap(qryDocList)
  return (X, queries, index_map)
 
def pointwise_learning(X, y):
  model = linear_model.LinearRegression()
  model.fit(X, y)
  return model

def pointwise_testing(X, model):
  y = model.predict(X)
  return y

##############################
##### Pair-wise approach #####
##############################
def pairwise_train_features(train_data_file, train_rel_file):
  queries, documents = read_feature_file(train_data_file)
  features, qryDocList = build_features(queries, documents)
  labels = build_labels(train_rel_file, qryDocList)
  X, y = build_pair_data(queries, features, qryDocList, labels)
  return (X, y)

def pairwise_test_features(test_data_file):
  queries, documents = read_feature_file(test_data_file)
  X, qryDocList = build_features(queries, documents)
  X = preprocessing.scale(X)
  queries, index_map = build_indexmap(qryDocList)
  return (X, queries, index_map)

def pairwise_learning(X, y):
  model = svm.SVC(kernel='linear', C=1.0)
  model.fit(X, y)
  return model

def pairwise_testing(X, model):
  y = []
  weights = model.coef_
  for ii in range(len(X)):
    y.append(np.dot(X[ii], weights[0]))
  return y

##############################
##### Task3 more feature #####
##############################
def task3_train_features(train_data_file, train_rel_file):
  queries, documents = read_feature_file(train_data_file)
  features, qryDocList = build_rich_features(queries, documents)
  labels = build_labels(train_rel_file, qryDocList)
  X, y = build_pair_data(queries, features, qryDocList, labels)
  return (X, y)

def task3_test_features(test_data_file):
  queries, documents = read_feature_file(test_data_file)
  X, qryDocList = build_rich_features(queries, documents)
  X = preprocessing.scale(X)
  queries, index_map = build_indexmap(qryDocList)
  return (X, queries, index_map)

def task3_learning(X, y):
  model = svm.SVC(kernel='linear', C=1.0)
  model.fit(X, y)
  return model

def task3_testing(X, model):
  y = []
  weights = model.coef_
  for ii in range(len(X)):
    y.append(np.dot(X[ii], weights[0]))
  return y

##############################
##### Extra Credit #####
##############################
def extra_train_features(train_data_file, train_rel_file):
  queries, documents = read_feature_file(train_data_file)
  X, qryDocList = build_features(queries, documents)
  y = build_labels(train_rel_file, qryDocList)
  return (X, y)
 
def extra_test_features(test_data_file):
  queries, documents = read_feature_file(test_data_file)
  X, qryDocList = build_features(queries, documents)
  queries, index_map = build_indexmap(qryDocList)
  return (X, queries, index_map)

def extra_learning(X, y):
  model = linear_model.LinearRegression()
  model.fit(X, y)
  return model

def extra_testing(X, model):
  y = model.predict(X)
  return y

####################
##### Training #####
####################
def train(train_data_file, train_rel_file, task):
  sys.stderr.write('\n## Training with feature_file = %s, rel_file = %s ... \n' % (train_data_file, train_rel_file))
  
  if task == 1:
    (X, y) = pointwise_train_features(train_data_file, train_rel_file)
    model = pointwise_learning(X, y)
  elif task == 2:
    (X, y) = pairwise_train_features(train_data_file, train_rel_file)
    model = pairwise_learning(X, y)
  elif task == 3: 
    (X, y) = task3_train_features(train_data_file, train_rel_file)
    model = task3_learning(X, y)
  elif task == 4: 
    # Extra credit 
    (X, y) = extra_train_features(train_data_file, train_rel_file)
    model = extra_learning(X, y)    
  else:
    (X, y) = pointwise_train_features(train_data_file, train_rel_file)
    model = pointwise_learning(X, y)
  # some debug output
  weights = model.coef_
  print >> sys.stderr, "Weights:", str(weights)

  return model 

###################
##### Testing #####
###################
def test(test_data_file, model, task):
  sys.stderr.write('\n## Testing with feature_file = %s ... \n' % (test_data_file))

  if task == 1:
    (X, queries, index_map) = pointwise_test_features(test_data_file)
    y = pointwise_testing(X, model)
  elif task == 2:
    (X, queries, index_map) = pairwise_test_features(test_data_file)
    y = pairwise_testing(X, model)
  elif task == 3: 
    (X, queries, index_map) = task3_test_features(test_data_file)
    y = task3_testing(X, model)
  elif task == 4:
    # Extra credit 
    (X, queries, index_map) = extra_test_features(test_data_file)
    y = extra_testing(X, model)
  else:
    (X, queries, index_map) = pointwise_test_features(test_data_file)
    y = pointwise_testing(X, model)
  # Print results
  print_results(queries, index_map, y)
  #print_to_file_results(queries, index_map, y)
  

if __name__ == '__main__':

  '''
  sys.stderr.write('# Input arguments: %s\n' % str(sys.argv))
  
  if len(sys.argv) != 5:
    print >> sys.stderr, "Usage:", sys.argv[0], "train_data_file train_rel_file test_data_file task"
    sys.exit(1)
  
  
  train_data_file = sys.argv[1]
  train_rel_file = sys.argv[2]
  test_data_file = sys.argv[3]
  task = int(sys.argv[4])
  print >> sys.stderr, "### Running task", task, "..."
  '''
  
  task = 1
  train_data_file = 'queryDocTrainData.train'
  train_rel_file = 'queryDocTrainRel.train'
  test_data_file = 'queryDocTrainData.dev'
  
  
  model = train(train_data_file, train_rel_file, task)
  test(test_data_file, model, task)
