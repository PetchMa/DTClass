#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from support_functions import generate_features_targets, plot_confusion_matrix, calculate_accuracy


# Spliting data into training and test
def splitdata_train_test(data, fraction_training):
    np.random.shuffle(data)
    split_index = int(fraction_training *len(data))
    return data[:split_index], data[split_index:]

#Label the data 
def generate_features_targets(data):
    targets = data['class']
    #
    features = np.empty(shape=(len(data), 13))
    #Colour Index 
    features[:, 0] = data['u-g']
    features[:, 1] = data['g-r']
    features[:, 2] = data['r-i']
    features[:, 3] = data['i-z']
    #eccentricity 
    features[:, 4] = data['ecc']
    #Movement 
    features[:, 5] = data['m4_u']
    features[:, 6] = data['m4_g']
    features[:, 7] = data['m4_r']
    features[:, 8] = data['m4_i']
    features[:, 9] = data['m4_z']
    #Concentration Ratio of colour
    features[:, 10] = data['petroR50_u']/data['petroR90_u']
    features[:, 11] = data['petroR50_r']/data['petroR90_r']
    features[:, 12] = data['petroR50_z']/data['petroR90_z']

    return features, targets
def splitdata_train_test(data, fraction_training):
  np.random.seed(0)
  np.random.shuffle(data)
  training_rows = math.floor(len(data)*fraction_training)
  training_set = data[0:training_rows]
  testing_set = data[training_rows:len(data)]
  
  return (training_set, testing_set)

# copy your generate_features_targets function here
def generate_features_targets(data):
  # complete the function by calculating the concentrations

  targets = data['class']

  features = np.empty(shape=(len(data), 13))
  features[:, 0] = data['u-g']
  features[:, 1] = data['g-r']
  features[:, 2] = data['r-i']
  features[:, 3] = data['i-z']
  features[:, 4] = data['ecc']
  features[:, 5] = data['m4_u']
  features[:, 6] = data['m4_g']
  features[:, 7] = data['m4_r']
  features[:, 8] = data['m4_i']
  features[:, 9] = data['m4_z']

  # fill the remaining 3 columns with concentrations in the u, r and z filters
  # concentration in u filter
  features[:, 10] = data['petroR50_u']/data['petroR90_u']
  # concentration in r filter
  features[:, 11] = data['petroR50_r']/data['petroR90_r']
  # concentration in z filter
  features[:, 12] = data['petroR50_z']/data['petroR90_z']

  return features, targets

#Calculates training accuracy 
def calculate_accuracy(predicted, actual):
  correct = 0
  # iterate over the list
  for p,a in zip(predicted,actual):
    if p == a:
       correct += 1
  accuracy = correct / len(actual)
  return accuracy

# Training a decision tree classifier
def dtc_predict_actual(data):
  # split the data into training and testing 
  training_set, testing_set = splitdata_train_test(data, 0.7)
  # generate the feature and targets for the training and test sets
  features_training, targets_training = generate_features_targets(training_set)
  features_testing, targets_testing = generate_features_targets(testing_set)
  # instantiate a decision tree classifier
  dtc = DecisionTreeClassifier()
  # train the classifier 
  dtc.fit(features_training, targets_training)
  # get predictions 
  predictions = dtc.predict(features_testing)
  # return the predictions and targets
  return predictions, targets_testing

#Random Forest 
def rf_predict_actual(data, n_estimators):

  features, targets = generate_features_targets(data)

  # instantiate a random forest classifier
  rfc = RandomForestClassifier(n_estimators=n_estimators)
  
  # get predictions using 10-fold cross validation with cross_val_predict
  predicted = cross_val_predict(rfc, features, targets, cv=10)

  # return the predictions and their actual classes
  return predicted, targets

#Runs the code
if __name__ == "__main__":
  data = np.load('galaxy_catalogue.npy')

  # get the predicted and actual classes
  number_estimators = 50              
  # Number of trees
  predicted, actual = rf_predict_actual(data, number_estimators)

  # calculate the model score using your function
  accuracy = calculate_accuracy(predicted, actual)
  print("Accuracy score:", accuracy)

  # calculate the models confusion matrix using sklearns confusion_matrix function
  class_labels = list(set(actual))
  model_cm = confusion_matrix(y_true=actual, y_pred=predicted, labels=class_labels)

  # plot the confusion matrix using the provided functions.
  plt.figure()
  plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
  plt.show()

