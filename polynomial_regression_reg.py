#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
print ''

# Input features: columns 8-40.
# Training data: countries 1-100 (Afghanistan to Luxembourg).
# Testing data: countries 101-195 (Madagascar to Zimbabwe).


targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100;
# trainin set input
x_train = x[0:N_TRAIN,:]

# test set input
#x_test = x[N_TRAIN:,:]

# training set target values
t_train = targets[0:N_TRAIN]

# test set target values
#t_test = targets[N_TRAIN:]

# set lambda
lambda_set = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# set w0 with coefficient 1 to training set
w0_term_train = np.ones(shape=(100,1))
# set w0 with coefficient 1 to test set
#w0_term_test = np.ones(shape=(95,1))

phi = w0_term_train
#test_set = w0_term_test


# design matrix, phi for training set, test_set for testing set
for degree_index in range(2):
  degree = degree_index + 1
  phi = np.concatenate((phi, np.power(x_train, degree)), axis=1)
  #test_set = np.concatenate((test_set, np.power(x_test, degree)), axis=1)



# number of columns in the phi
coefficient_M = np.shape(phi)[1]
identity_matrix = np.identity(coefficient_M)

#print(coefficient_M)
#print(np.shape(identity_matrix))

# the size of partition in the 10-fold cross-validation
partition = len(t_train) / 10

lambda_error = []
# for each lambda
for lambda_index in range(len(lambda_set)):
  lambda_value = lambda_set[lambda_index]
  validation_error = []
  
  # for each validation set in the 10-fold cross-validation
  for validation_index in range(10):
    validation_set = validation_index * partition # start index of each partition
    validation_phi = phi[validation_set:validation_set+partition,:]
    validation_t = t_train[validation_set:validation_set+partition,:]
    training_phi = np.concatenate((phi[0:validation_set,:], phi[validation_set+partition:,:]), axis=0)
    training_t = np.concatenate((t_train[0:validation_set,:], t_train[validation_set+partition:,:]), axis=0)
    w = np.dot(np.dot(np.linalg.inv(np.add(np.dot(lambda_value, identity_matrix), np.dot(training_phi.transpose(), training_phi))), training_phi.transpose()), training_t)
    validation_partition_error = np.sqrt(0.5 * np.sum(np.square(np.dot(validation_phi, w) - validation_t)) * 2 / len(validation_t))

    validation_error.append(validation_partition_error)
    #print(validation_partition_error)

  lambda_error.append(np.mean(validation_error))


print(lambda_error)
#lambda_plot = np.split(lambda_value, 1)
#error_plot = np.split(lambda_error, 1)
#print(lambda_set[1:8])
#print(lambda_error)

#print(lambda_plot)
#print(error_plot)


# Produce a plot of results.
plt.semilogx(lambda_set[1:8], lambda_error[1:8])
plt.xlabel('Lambda Value on log Scale')
plt.ylabel('Average Validation Set Error')
plt.title('Average Validation Set Error vs. Lambda Value on log Scale ')
#plt.xticks(np.log10(lambda_set[1:8]), lambda_set[1:8])
plt.axhline(y=lambda_error[0], linewidth=2, color = 'g', label = '')
plt.legend(['Average validation set error', 'Unregularized result, Lambda = 0'])
plt.show()
  

