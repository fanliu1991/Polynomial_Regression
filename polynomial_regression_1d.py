#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

# Input features: columns 8-40.
# Training data: countries 1-100 (Afghanistan to Luxembourg).
# Testing data: countries 101-195 (Madagascar to Zimbabwe).


targets = values[:,1]
x = values[:,7:]

#print(x[:,0])
N_TRAIN = 100;
# trainin set input
x_train = x[0:N_TRAIN,:8]

# test set input
x_test = x[N_TRAIN:,:8]

# training set target values
t_train = targets[0:N_TRAIN]

# test set target values
t_test = targets[N_TRAIN:]

# 4.2 Polynomial Regression
#1.
# set w0 with coefficient 1 to training set
w0_term_train = np.ones(shape=(len(t_train),1))
# set w0 with coefficient 1 to test set
w0_term_test = np.ones(shape=(len(t_test),1))


w_set = []
train_err = {}
test_err = {}

print ''
for feature_index in range(8):
  phi = w0_term_train
  test_set = w0_term_test
  
  for degree_index in range(3):
    degree = degree_index + 1
    phi = np.concatenate((phi, np.power(x_train[0:N_TRAIN, feature_index], degree)), axis=1)
    test_set = np.concatenate((test_set, np.power(x_test[0:N_TRAIN, feature_index], degree)), axis=1)
    
  w = np.dot(np.linalg.pinv(phi), t_train)
  w_set.append(w)
  training_error = np.sqrt(0.5 * np.sum(np.square(np.dot(phi, w) - t_train)) * 2 / len(t_train))
  test_error = np.sqrt(0.5 * np.sum(np.square(np.dot(test_set, w) - t_test)) * 2 / len(t_test))
  train_err[feature_index + 1] = training_error
  test_err[feature_index + 1] = test_error
  print(w)
  print(training_error)
  print(test_error)
  print('')
  print('')
  



# Produce a plot of results.
#fig, ax = plt.subplots()
index = np.arange(len(train_err))
bar_width = 0.35

rects1 = plt.bar(index + bar_width, train_err.values(), bar_width, color='b', label='Train')
rects2 = plt.bar(index + bar_width*2, test_err.values(), bar_width, color='g', label='Test')
plt.xlabel('Features')
plt.ylabel('RMS')
plt.legend(['Training error', 'Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xticks(index + bar_width*2, ('Total population \n(thousands) 2011', 'Annual no. of births \n(thousands) 2011', 'Annual no. of under-5 \ndeaths (thousands) 2011', 'GNI per capita \n(US$) 2011', 'Life expectancy at \nbirth (years) 2011', 'Total adult literacy \nrate (%) 2007_2011', 'Primary school net enrolment \nratio (%) 2008_2011', 'Low birthweight \n(%) 2007_2011'))



# 4.2-2 GNI, Life expectancy, literacy
# Select feature, GNI, Life expectancy, literacy.
GNI = x[:,3]
GNI_train = x[0:N_TRAIN,3]
GNI_test = x[N_TRAIN:,3]

life_expect = x[:,4]
life_expect_train = x[0:N_TRAIN,4]
life_expect_test = x[N_TRAIN:,4]

literacy = x[:,5]
literacy_train = x[0:N_TRAIN,5]
literacy_test = x[N_TRAIN:,5]

# regression model of GNI
GNI_ev = np.linspace(np.asscalar(min(GNI)), np.asscalar(max(GNI)), num=500)
GNI_polynomial = np.random.random_sample(GNI_ev.shape)
GNI_polynomial = np.asscalar(w_set[3][0]) + np.asscalar(w_set[3][1])*GNI_ev + np.asscalar(w_set[3][2])*GNI_ev*GNI_ev + np.asscalar(w_set[3][3])*GNI_ev*GNI_ev*GNI_ev

# regression model of life expectancy
life_expect_ev = np.linspace(np.asscalar(min(life_expect)), np.asscalar(max(life_expect)), num=500)
life_expect_polynomial = np.random.random_sample(life_expect_ev.shape)
life_expect_polynomial = np.asscalar(w_set[4][0]) + np.asscalar(w_set[4][1])*life_expect_ev + np.asscalar(w_set[4][2])*life_expect_ev*life_expect_ev + np.asscalar(w_set[4][3])*life_expect_ev*life_expect_ev*life_expect_ev

# regression model of literacy
literacy_ev = np.linspace(np.asscalar(min(literacy)), np.asscalar(max(literacy)), num=500)
literacy_polynomial = np.random.random_sample(literacy_ev.shape)
literacy_polynomial = np.asscalar(w_set[5][0]) + np.asscalar(w_set[5][1])*literacy_ev + np.asscalar(w_set[5][2])*literacy_ev*literacy_ev + np.asscalar(w_set[5][3])*literacy_ev*literacy_ev*literacy_ev



fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax1.plot(GNI_ev,GNI_polynomial,'r.-')
ax1.plot(GNI_train, t_train, 'bo')
ax1.plot(GNI_test,t_test,'gv')
ax1.legend(['Regression Estimate', 'Training data point', 'Test data point'], loc = 'lower left', fontsize = 12)
ax1.set_title('A visualization of the fits for degree 3 polynomials for GNI per capita (US$) 2011')


ax2 = fig.add_subplot(3,1,2)
ax2.plot(life_expect_ev,life_expect_polynomial,'r.-')
ax2.plot(life_expect_train, t_train, 'bo')
ax2.plot(life_expect_test,t_test,'gv')
ax2.legend(['Regression Estimate', 'Training data point', 'Test data point'], loc = 'lower left', fontsize = 12)
ax2.set_title('A visualization of the fits for degree 3 polynomials for life expectancy at birth (years) 2011')


ax3 = fig.add_subplot(3,1,3)
ax3.plot(literacy_ev,literacy_polynomial,'r.-')
ax3.plot(literacy_train, t_train, 'bo')
ax3.plot(literacy_test,t_test,'gv')
ax3.legend(['Regression Estimate', 'Training data point', 'Test data point'], loc = 'lower left', fontsize = 12)
ax3.set_title('A visualization of the fits for degree 3 polynomials for total adult literacy rate (%) 2007_2011')


plt.show()
