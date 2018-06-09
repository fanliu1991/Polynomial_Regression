#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# Select feature GNI
GNI = x[:,3]
GNI_train = x[0:N_TRAIN,3]
GNI_test = x[N_TRAIN:,3]

# set w0 with coefficient 1 to training set
w0_term_train = np.ones(shape=(len(t_train),1))
# set w0 with coefficient 1 to test set
w0_term_test = np.ones(shape=(len(t_test),1))


# sigmoid basis functions, with u = 100 and s = 2000.0
GNI_train_u100 = 1 / (1 + np.exp((100 - GNI_train) / 2000))
GNI_test_u100 = 1 / (1 + np.exp((100 - GNI_test) / 2000))
# # sigmoid basis functions, with u = 10000 and s = 2000.0
GNI_train_u10000 = 1 / (1 + np.exp((10000 - GNI_train) / 2000))
GNI_test_u10000 = 1 / (1 + np.exp((10000 - GNI_test) / 2000))
# design matrix
phi = np.concatenate((w0_term_train, GNI_train_u100), axis=1)
phi = np.concatenate((phi, GNI_train_u10000), axis=1)
test_set = np.concatenate((w0_term_test, GNI_test_u100), axis=1)
test_set = np.concatenate((test_set, GNI_test_u10000), axis=1)
# w = (phi_T * phi)^-1 * phi_T * t
w = np.dot(np.linalg.pinv(phi), t_train)
# training error
training_error = np.sqrt(0.5 * np.sum(np.square(np.dot(phi, w) - t_train)) * 2 / len(t_train))
# test error
test_error = np.sqrt(0.5 * np.sum(np.square(np.dot(test_set, w) - t_test)) * 2 / len(t_test))
print(training_error)
print(test_error)


GNI_ev = np.linspace(np.asscalar(min(GNI)), np.asscalar(max(GNI)), num=1000)
# regression model of GNI, sigmoid basis functions with u = 100, u = 10000 and s = 2000.0
GNI_ev_u100 = 1 / (1 + np.exp((100 - GNI_ev) / 2000))
GNI_ev_u10000 = 1 / (1 + np.exp((10000 - GNI_ev) / 2000))
GNI_polynomial = np.random.random_sample(GNI_ev.shape)
GNI_polynomial = np.asscalar(w[0]) + np.asscalar(w[1])*GNI_ev_u100 + np.asscalar(w[2])*GNI_ev_u10000


plt.plot(GNI_ev,GNI_polynomial,'r.-')
plt.plot(GNI_train, t_train, 'bo')
plt.plot(GNI_test,t_test,'gv')
plt.legend(['Regression Estimate', 'Training data point', 'Test data point'], loc = 'upper right', fontsize = 12)
plt.title('A visualization of the fits for GNI with sigmoid basis functions, u = 100, u = 10000 and s = 2000.0')



'''
# sigmoid basis functions, with u = 100 and s = 2000.0
GNI_train_u100 = 1 / (1 + np.exp((100 - GNI_train) / 2000))
GNI_test_u100 = 1 / (1 + np.exp((100 - GNI_test) / 2000))
# design matrix
phi_u100 = np.concatenate((w0_term_train, GNI_train_u100), axis=1)
testset_u100 = np.concatenate((w0_term_test, GNI_test_u100), axis=1)
# w = (phi_T * phi)^-1 * phi_T * t
w_u100 = np.dot(np.linalg.pinv(phi_u100), t_train)
# training error for u = 100
training_error_u100 = np.sqrt(0.5 * np.sum(np.square(np.dot(phi_u100, w_u100) - t_train)) * 2 / len(t_train))
# test error for u = 100
test_error_u100 = np.sqrt(0.5 * np.sum(np.square(np.dot(testset_u100, w_u100) - t_test)) * 2 / len(t_test))
print(training_error_u100)
print(test_error_u100)
#print(w_u100)
print('')
print('')

# # sigmoid basis functions, with u = 10000 and s = 2000.0
GNI_train_u10000 = 1 / (1 + np.exp((10000 - GNI_train) / 2000))
GNI_test_u10000 = 1 / (1 + np.exp((10000 - GNI_test) / 2000))
# design matrix
phi_u10000 = np.concatenate((w0_term_train, GNI_train_u10000), axis=1)
testset_u100 = np.concatenate((w0_term_test, GNI_test_u10000), axis=1)
# w = (phi_T * phi)^-1 * phi_T * t
w_u10000 = np.dot(np.linalg.pinv(phi_u10000), t_train)
# training error for u = 10000
training_error_degree_2 = np.sqrt(0.5 * np.sum(np.square(np.dot(phi_u10000, w_u10000) - t_train)) * 2 / len(t_train))
# test error for u = 10000
test_error_degree_2 = np.sqrt(0.5 * np.sum(np.square(np.dot(testset_u100, w_u10000) - t_test)) * 2 / len(t_test))
print(training_error_degree_2)
print(test_error_degree_2)
#print(w_u10000)
print('')
print('')


GNI_ev = np.linspace(np.asscalar(min(GNI)), np.asscalar(max(GNI)), num=1000)
# regression model of GNI, sigmoid basis functions with u = 100 and s = 2000.0
GNI_ev_u100 = 1 / (1 + np.exp((100 - GNI_ev) / 2000))
GNI_polynomial_u100 = np.random.random_sample(GNI_ev_u100.shape)
GNI_polynomial_u100 = 212.8847733 + (-204.4823977)*GNI_ev_u100
# regression model of GNI, sigmoid basis functions with u = 10000 and s = 2000.0
GNI_ev_u10000 = 1 / (1 + np.exp((10000 - GNI_ev) / 2000))
GNI_polynomial_u10000 = np.random.random_sample(GNI_ev_u10000.shape)
GNI_polynomial_u10000 = 61.68903018 + (-58.18409323)*GNI_ev_u10000


# plot of the fits
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(GNI_ev,GNI_polynomial_u100,'r.-')
ax1.plot(GNI_train, t_train, 'bo')
ax1.plot(GNI_test,t_test,'gv')
ax1.legend(['Regression Estimate', 'Training data point', 'Test data point'], loc = 'upper right', fontsize = 12)
ax1.set_title('A visualization of the fits for GNI with sigmoid basis functions, u = 100 and s = 2000.0')

ax2 = fig.add_subplot(2,1,2)
ax2.plot(GNI_ev,GNI_polynomial_u10000,'r.-')
ax2.plot(GNI_train, t_train, 'bo')
ax2.plot(GNI_test,t_test,'gv')
ax2.legend(['Regression Estimate', 'Training data point', 'Test data point'], loc = 'upper right', fontsize = 12)
ax2.set_title('A visualization of the fits for GNI with sigmoid basis functions, u = 10000 and s = 2000.0')


plt.plot(GNI_ev,GNI_polynomial_u100,'r.-')
plt.plot(GNI_train, t_train, 'bo')
plt.plot(GNI_test,t_test,'gv')
plt.legend(['Regression Estimate', 'Training data point', 'Test data point'], loc = 'upper right', fontsize = 11)
plt.title('A visualization of the fits for GNI with sigmoid basis functions, u = 100 and s = 2000.0')


plt.plot(GNI_ev_u10000,GNI_polynomial_u10000,'r.-')
plt.plot(GNI_train, t_train, 'bo')
plt.plot(GNI_test,t_test,'gv')
plt.legend(['Regression Estimate', 'Training data point', 'Test data point'], loc = 'upper right', fontsize = 11)
plt.title('A visualization of the fits for GNI with sigmoid basis functions, u = 10000 and s = 2000.0')
'''

plt.show()

