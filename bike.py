from __future__ import print_function, division
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, minmax_scale
def mae(y, prediction):
    return np.mean(np.abs(y - prediction))

# Read in full data set
X = pd.read_csv(
    'Bike-Sharing-Dataset/hour.csv', sep=',',
    dayfirst=False,
    # parse_dates={'datetime' : ['dteday', 'hr']},
    #names=['yr', 'mnth', 'holiday', 'workingday', 'weathersit', 'weekday', 'temp', 'atemp', 'hum', 'windspeed']
    )

# Add time series
X = X.sort_values(['dteday', 'hr'])
# Add ewma over same hour
cols = ['hr', 'weekday']
#cols = ['hr']
gb = X[['cnt'] + cols].groupby(cols)

for halflife in [1.5]:
    def ewma(X):
        X = X.shift()
        X = pd.DataFrame(pd.ewma(X['cnt'], halflife=halflife))
        return X
    x = gb.transform(ewma)
    x = x.fillna(0)
    print("MAE ewma", halflife, mae(X['cnt'], x['cnt']))

#X['cnt_ewma'] = x['cnt']
#from IPython import embed
#embed()


def train_test_split(X):
    n = len(X)
    index = np.random.permutation(np.arange(0, n))

    X_train = X.iloc[index[n // 10:]]
    X_test = X.iloc[index[:n // 10]]
    assert len(X) == len(X_train) + len(X_test)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    return X_train, X_test

df_train, df_test = train_test_split(X)


print("MAE trivial", mae(df_test['cnt'], np.mean(df_train['cnt'])))
print("MAE trivial median", mae(df_test['cnt'], np.median(df_train['cnt'])))

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
est = RandomForestRegressor(n_estimators=2000, n_jobs=4)
#est = GradientBoostingRegressor(n_estimators=2000)
def extract_target(X):
    y = np.asarray(X['cnt'])
    Xnp = X.copy()
    del Xnp['cnt']
    del Xnp['dteday']
    del Xnp['casual']
    del Xnp['registered']
    del Xnp['instant']
    columns = Xnp.columns
    Xnp = np.asarray(Xnp)
    return Xnp, y, columns

X_train, y_train, columns = extract_target(df_train)
#est.fit(X_train, y_train - df_train['cnt_ewma'])
est.fit(X_train, y_train)

X_test, y_test, _ = extract_target(df_test)

prediction = est.predict(X_test)
#prediction += df_test['cnt_ewma']
importances = pd.Series(est.feature_importances_, index=columns)
importances.plot(kind='bar')
plt.show()
print("MAE trivial random forest", mae(df_test['cnt'], prediction))

# fit
df_test['pred'] = prediction
gb = df_test.groupby('dteday').mean()
gb['cnt'] .plot()
gb['pred'].plot()
plt.show()
sm = gb
sm['cnt_day'] = sm['cnt']
sm = sm.reset_index()

# predict
Xm = X.merge(sm[['cnt_day', 'dteday']], on=['dteday'], how='left')
