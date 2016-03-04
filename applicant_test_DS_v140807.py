
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, minmax_scale

# Read in full data set
X = pd.read_csv(
    'Bike-Sharing-Dataset/hour.csv', sep=',',
    dayfirst=False,
    # parse_dates={'datetime' : ['dteday', 'hr']},
    #names=['yr', 'mnth', 'holiday', 'workingday', 'weathersit', 'weekday', 'temp', 'atemp', 'hum', 'windspeed']
    )

# Take 10% randomly selected events
Y = X[X['yr'] == 1].sample(frac=1)

#X['date'] = X['dteday'] + ':' + X['hr'].map(str)

# TODO: check for outliers, missing data, transform variables (binning, logarith, rescale, etc)
# Data preprocessing required: Remove outliers, check for missing data, etc

# Improve plots: binning, logscale, variable transformation

# Verify obvious correlations (better compute covariance matrix)

# validate hypothesis using correlation factors of continuous variables

# Check correlation with time: peaks when office-hour traffic is high
# X.plot(x='hr', y=['cnt'], style='.')

Y['ncasual'] = minmax_scale(Y['casual'])
Y['nregistered'] = minmax_scale(Y['registered'])


# Registerd users contribute more during office-hour traffic
# while casual users have a broad peak starting after late breakfast and levelling when sun sets
Y.plot(x='hr', y=['ncasual', 'nregistered'], style=['.', '+'])
# X.plot(x='mnth', y=['ncasual', 'nregistered'], style=['.', '+'])

X[X['windspeed'] > 0].plot(x='windspeed', y=['registered', 'casual'], kind='hist', subplots='True', sharey=True,
                           logy=True)
X.plot(x='windspeed', y='casual', style='.')



# X.plot(x='mnth', y='cnt', style='.')



# X.plot(x='datetime', y=['cnt'], style='.')
#X.plot(x='atemp', y='cnt', style='.')


#
#
# from sklearn import svm, preprocessing
# scaler = preprocessing.StandardScaler
#
# clf = svm.SVC(gamma='auto', C=1, kernel='poly')
#
# predictors = ['mnth', 'hr', 'weekday', 'workingday']
# # predictors = ['yr', 'mnth', 'weekday', 'hr', 'weathersit', 'atemp']
# target = 'registered'
#
# # Training data
# data_train = Y[predictors].values
#
# # Rescale data
# scaler = preprocessing.RobustScaler().fit(data_train)
# data_train = scaler.transform(data_train)
# target_train = Y[target].values
# # target_train = np.log(target_train + 1)
#
# # Prediction
# data_test = X[predictors][:500].values
# data_test = scaler.transform(data_test)
#
# # target_test = np.log(X['cnt'][:500].values + 1)
# target_test = X[target][:500].values
# # target_test = np.log(target_test + 1)
#
# clf.fit(data_train, target_train)
#
# target_predict = clf.predict(data_test)
#
#
# print('score:', clf.score(data_test, target_test))
#
# print(target_test[:10])
# print(target_predict[:10])
#
# mae = np.abs(target_test - target_predict).mean()
# rmsle = np.sqrt(1. / target_test.size * np.sum(
#         (np.log(target_predict + 1) - np.log(target_test + 1))**2))
#
# print('MAE: {}'.format(mae))
# print('relative MAE: {}'.format(mae/np.mean(target_test)))
# print('RMSLE: {}'.format(rmsle))
#
plt.show()
#







# from scipy.ndimage import convolve
# from sklearn import linear_model, datasets, metrics
# from sklearn.cross_validation import train_test_split
# from sklearn.neural_network import BernoulliRBM
# from sklearn.pipeline import Pipeline

#
# X = pd.read_csv(
#     'Bike-Sharing-Dataset/day.csv', sep=',',
#     names=['yr', 'mnth', 'holiday', 'workingday', 'weathersit', 'weekday', 'temp', 'atemp', 'hum', 'windspeed']
#     ).values[1:, :].astype(float)
#
# # 0-1 scaling
# X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)
#
# Y = pd.read_csv(
#     'Bike-Sharing-Dataset/day.csv', sep=',',
#     names=['cnt']).values[1:, 0].astype(float)
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                     test_size=0.1,
#                                                     random_state=0)
#
# # Models we will use
# logistic = linear_model.LogisticRegression()
# rbm = BernoulliRBM(random_state=0, verbose=True)
#
# classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])


# ##############################################################################
# # Training
#
# # Hyper-parameters. These were set by cross-validation,
# # using a GridSearchCV. Here we are not performing cross-validation to
# # save time.
# rbm.learning_rate = 0.06
# rbm.n_iter = 20
# # More components tend to give better prediction performance, but larger
# # fitting time
# rbm.n_components = 100
# logistic.C = 6000.0
#
# # Training RBM-Logistic Pipeline
# classifier.fit(X_train, Y_train)
#
# # Training Logistic regression
# logistic_classifier = linear_model.LogisticRegression(C=100.0)
# logistic_classifier.fit(X_train, Y_train)
#
# ###############################################################################
# # Evaluation
#
# print()
# print("Logistic regression using RBM features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,
#         classifier.predict(X_test))))
#
# print("Logistic regression using raw pixel features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,
#         logistic_classifier.predict(X_test))))
#
