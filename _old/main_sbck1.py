import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import SBCK


obsv_p = pd.read_csv('./data/observations_precipitation.csv')
obsv_t = pd.read_csv('./data/observations_temperature.csv')
pred_p = pd.read_csv('./data/predictions_precipitation.csv')
pred_t = pd.read_csv('./data/predictions_temperature.csv')

obsv = pd.merge(obsv_p, obsv_t, on=['year', 'month', 'day'], suffixes=("_p", "_t"))
pred = pd.merge(pred_p, pred_t, on=['year', 'month', 'day'], suffixes=("_p", "_t"))

df = pd.merge(obsv, pred, on=['year', 'month', 'day'], suffixes=("_obsv", "_pred"))

df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df['season'] = (df['month'] // 3) % 4

df_calibration = df[(df['year'] >= 1976) & (df['year'] <= 1995)]
df_validation = df[(df['year'] >= 1996) & (df['year'] <= 2005)]

Y_calibration = df_calibration[['value_p_obsv', 'value_t_obsv']].values
X_calibration = df_calibration[['value_p_pred', 'value_t_pred']].values
Y_validation = df_validation[['value_p_obsv', 'value_t_obsv']].values
X_validation = df_validation[['value_p_pred', 'value_t_pred']].values

def calculatebias(X, Y, Z, daterange):
    B_original = X - Y
    B_corrected = Z - Y
    df_bias = pd.DataFrame(np.hstack([B_original, B_corrected]), columns=['bias_original_p', 'bias_original_t', 'bias_corrected_p', 'bias_corrected_t'], index=daterange)
    df_bias = df_bias.groupby(df_bias.index.month).mean()
    df_bias = df_bias.set_index(df_bias.index.map(lambda i: calendar.month_name[i]))
    return df_bias

bcmethod = SBCK.QM()

bcmethod.fit(Y_calibration, X_calibration)

Z_calibration = bcmethod.predict(X_calibration)
df_bias_calibration = calculatebias(X_calibration, Y_calibration, Z_calibration, df_calibration['date'])

Z_validation = bcmethod.predict(X_validation)
df_bias_validation = calculatebias(X_validation, Y_validation, Z_validation, df_validation['date'])

df_bias_merged = pd.merge(df_bias_calibration, df_bias_validation, on='date', suffixes=("_cal", "_val"))

def plotbias(df, title):
    minvalue = df.min(axis=None)
    maxvalue = df.max(axis=None)
    limitvalue = max(abs(minvalue), abs(maxvalue)) * 1.2
    df.plot.bar(title=title, ylim=(-limitvalue, limitvalue))

plotbias(df_bias_merged.filter(like='_p'), title='Precipitation bias')
plotbias(df_bias_merged.filter(like='_t'), title='Temperature bias')

plt.show()