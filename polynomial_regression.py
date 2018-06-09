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
normalize_x = a1.normalize_data(x)

N_TRAIN = 100;
# trainin set input
x_train = x[0:N_TRAIN,:]
normalize_x_train = normalize_x[0:N_TRAIN,:]

# test set input
x_test = x[N_TRAIN:,:]
normalize_x_test = normalize_x[N_TRAIN:,:]

# training set target values
t_train = targets[0:N_TRAIN]

# test set target values
t_test = targets[N_TRAIN:]

# 4.2 Polynomial Regression
#1.
# set w0 with coefficient 1 to training set
w0_term_train = np.ones(shape=(100,1))
# set w0 with coefficient 1 to test set
w0_term_test = np.ones(shape=(95,1))


'''

# Degree = 1
phi_degree_1 = np.concatenate((w0_term_train, x_train), axis=1)
testset_degree_1 = np.concatenate((w0_term_test, x_test), axis=1)
# w = (phi_T * phi)^-1 * phi_T * t
w_degree_1 = np.dot(np.dot(np.linalg.inv(np.dot(phi_degree_1.transpose(), phi_degree_1)), phi_degree_1.transpose()), t_train)
# training error for degree = 1
training_error_degree_1 = np.sqrt(0.5 * np.sum(np.square(np.dot(phi_degree_1, w_degree_1) - t_train)) * 2 / len(t_train))
# test error for degree = 1
test_error_degree_1 = np.sqrt(0.5 * np.sum(np.square(np.dot(testset_degree_1, w_degree_1) - t_test)) * 2 / len(t_test))
print(training_error_degree_1)
print(test_error_degree_1)
print('')


# Degree = 2
phi_degree_2 = np.concatenate((phi_degree_1, np.power(x_train, 2)), axis=1)
testset_degree_2 = np.concatenate((testset_degree_1, np.power(x_test, 2)), axis=1)
# w = (phi_T * phi)^-1 * phi_T * t
w_degree_2 = np.dot(np.linalg.pinv(phi_degree_2), t_train)
# training error for degree = 2
training_error_degree_2 = np.sqrt(0.5 * np.sum(np.square(np.dot(phi_degree_2, w_degree_2) - t_train)) * 2 / len(t_train))
# test error for degree = 2
test_error_degree_2 = np.sqrt(0.5 * np.sum(np.square(np.dot(testset_degree_2, w_degree_2) - t_test)) * 2 / len(t_test))
print(training_error_degree_2)
print(test_error_degree_2)
print('')


# Degree = 3
phi_degree_3 = np.concatenate((phi_degree_2, np.power(x_train, 3)), axis=1)
testset_degree_3 = np.concatenate((testset_degree_2, np.power(x_test, 3)), axis=1)
# w = (phi_T * phi)^-1 * phi_T * t
#w_degree_3 = np.dot(np.dot(np.linalg.inv(np.dot(phi_degree_3.transpose(), phi_degree_3)), phi_degree_3.transpose()), t_train)
w_degree_3 = np.dot(np.linalg.pinv(phi_degree_3), t_train)
# training error for degree = 3
training_error_degree_3 = np.sqrt(0.5 * np.sum(np.square(np.dot(phi_degree_3, w_degree_3) - t_train)) * 2 / len(t_train))
# test error for degree = 3
test_error_degree_3 = np.sqrt(0.5 * np.sum(np.square(np.dot(testset_degree_3, w_degree_3) - t_test)) * 2 / len(t_test))
print(training_error_degree_3)
print(test_error_degree_3)
print('')


# Degree = 4
phi_degree_4 = np.concatenate((phi_degree_3, np.power(x_train, 4)), axis=1)
testset_degree_4 = np.concatenate((testset_degree_3, np.power(x_test, 4)), axis=1)
# w = (phi_T * phi)^-1 * phi_T * t
w_degree_4 = np.dot(np.linalg.pinv(phi_degree_4), t_train)
# training error for degree = 4
training_error_degree_4 = np.sqrt(0.5 * np.sum(np.square(np.dot(phi_degree_4, w_degree_4) - t_train)) * 2 / len(t_train))
# test error for degree = 4
test_error_degree_4 = np.sqrt(0.5 * np.sum(np.square(np.dot(testset_degree_4, w_degree_4) - t_test)) * 2 / len(t_test))
print(training_error_degree_4)
print(test_error_degree_4)
print('')

# Degree = 5
phi_degree_5 = np.concatenate((phi_degree_4, np.power(x_train, 5)), axis=1)
testset_degree_5 = np.concatenate((testset_degree_4, np.power(x_test, 5)), axis=1)
# w = (phi_T * phi)^-1 * phi_T * t
w_degree_5 = np.dot(np.linalg.pinv(phi_degree_5), t_train)
# training error for degree = 5
training_error_degree_5 = np.sqrt(0.5 * np.sum(np.square(np.dot(phi_degree_5, w_degree_5) - t_train)) * 2 / len(t_train))
# test error for degree = 5
test_error_degree_5 = np.sqrt(0.5 * np.sum(np.square(np.dot(testset_degree_5, w_degree_5) - t_test)) * 2 / len(t_test))
print(training_error_degree_5)
print(test_error_degree_5)
print('')


# Degree = 6
phi_degree_6 = np.concatenate((phi_degree_5, np.power(x_train, 6)), axis=1)
testset_degree_6 = np.concatenate((testset_degree_5, np.power(x_test, 6)), axis=1)
# w = (phi_T * phi)^-1 * phi_T * t
w_degree_6 = np.dot(np.linalg.pinv(phi_degree_6), t_train)
# training error for degree = 6
training_error_degree_6 = np.sqrt(0.5 * np.sum(np.square(np.dot(phi_degree_6, w_degree_6) - t_train)) * 2 / len(t_train))
# test error for degree = 3
test_error_degree_6 = np.sqrt(0.5 * np.sum(np.square(np.dot(testset_degree_6, w_degree_6) - t_test)) * 2 / len(t_test))
print(training_error_degree_6)
print(test_error_degree_6)
print('')

'''

phi = w0_term_train
test_set = w0_term_test

train_err = {}
test_err = {}

# non-normalize x
for index in range(6):
  degree = index + 1
  phi = np.concatenate((phi, np.power(x_train, degree)), axis=1)
  test_set = np.concatenate((test_set, np.power(x_test, degree)), axis=1)
  w = np.dot(np.linalg.pinv(phi), t_train)
  training_error = np.sqrt(0.5 * np.sum(np.square(np.dot(phi, w) - t_train)) * 2 / len(t_train))
  test_error = np.sqrt(0.5 * np.sum(np.square(np.dot(test_set, w) - t_test)) * 2 / len(t_test))
  train_err[index + 1] = training_error
  test_err[index + 1] = test_error
  #print(training_error)
  #print(test_error)
  #print('')
  #print('')


phi = w0_term_train
test_set = w0_term_test

normalize_train_err = {}
normalize_test_err = {}


# normalize x
for index in range(6):
  degree = index + 1
  phi = np.concatenate((phi, np.power(normalize_x_train, degree)), axis=1)
  test_set = np.concatenate((test_set, np.power(normalize_x_test, degree)), axis=1)
  w = np.dot(np.linalg.pinv(phi), t_train)
  training_error = np.sqrt(0.5 * np.sum(np.square(np.dot(phi, w) - t_train)) * 2 / len(t_train))
  test_error = np.sqrt(0.5 * np.sum(np.square(np.dot(test_set, w) - t_test)) * 2 / len(t_test))
  normalize_train_err[index + 1] = training_error
  normalize_test_err[index + 1] = test_error
  #print(training_error)
  #print(test_error)
  #print('')
  #print('')

'''
# Produce a plot of results.
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(train_err.keys(), train_err.values())
ax1.plot(test_err.keys(), test_err.values())
ax1.set_xlabel('Polynomial degree')
ax1.set_ylabel('RMS')
ax1.legend(['Training error', 'Test error'])
ax1.set_title('Fit with polynomials, no regularization, non-normalize data')

ax2 = fig.add_subplot(2,1,2)
ax2.plot(normalize_train_err.keys(), normalize_train_err.values())
ax2.plot(normalize_test_err.keys(), normalize_test_err.values())
ax2.set_ylim([-50,9000])
ax2.set_xlabel('Polynomial degree')
ax2.set_ylabel('RMS')
ax2.legend(['Training error', 'Test error'])
ax2.set_title('Fit with polynomials, no regularization, normalize data')
'''
'''
plt.plot(train_err.keys(), train_err.values())
plt.plot(test_err.keys(), test_err.values())
plt.xlabel('Polynomial degree')
plt.ylabel('RMS')
plt.legend(['Training error', 'Test error'])
plt.title('Fit with polynomials, no regularization, non-normalize data')

'''
plt.plot(normalize_train_err.keys(), normalize_train_err.values())
plt.plot(normalize_test_err.keys(), normalize_test_err.values())
plt.ylim([-50,9000])
plt.xlabel('Polynomial degree')
plt.ylabel('RMS')
plt.legend(['Training error', 'Test error'])
plt.title('Fit with polynomials, no regularization, normalize data')

plt.show()


