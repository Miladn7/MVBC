import calendar
import inspect
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

def calculatebias(X, Y, daterange):
    df_bias = pd.DataFrame(X - Y, columns=['bias_p', 'bias_t'], index=daterange)
    df_bias = df_bias.groupby(df_bias.index.month).mean()
    # df_bias = df_bias.set_index(df_bias.index.map(lambda i: calendar.month_name[i]))
    return df_bias

def apply(bcmethod):

    if (len(inspect.signature(bcmethod.fit).parameters) == 3):
        # fit bcmethod with calibration data
        bcmethod.fit(Y_calibration, X_calibration, X_validation)
        # correct bias of validation and calibration data using bcmethod
        Z_validation, Z_calibration = bcmethod.predict(X_validation, X_calibration)
    else:
        # fit bcmethod with calibration data
        bcmethod.fit(Y_calibration, X_calibration)
        # correct bias of calibration data using bcmethod
        Z_calibration = bcmethod.predict(X_calibration)
        # correct bias of validation data using bcmethod
        Z_validation = bcmethod.predict(X_validation)

    # calculate bias of corrected calibration data
    df_bias_calibration = calculatebias(Z_calibration, Y_calibration, df_calibration['date'])

    # calculate bias of corrected validation data
    df_bias_validation = calculatebias(Z_validation, Y_validation, df_validation['date'])

    return df_bias_calibration, df_bias_validation


bcmethods = [SBCK.QM(), SBCK.CDFt(), SBCK.OTC(), SBCK.R2D2(), SBCK.MBCn(), SBCK.QDM(), SBCK.MRec(), SBCK.ECBC()]
bcmethods = {bcmethod.__class__.__name__: bcmethod for bcmethod in bcmethods}

calibration_results = { 'original': calculatebias(X_calibration, Y_calibration, df_calibration['date']) }
validation_results = { 'original': calculatebias(X_validation, Y_validation, df_validation['date']) }

for name, bcmethod in bcmethods.items():
    bias_calibration, bias_validation = apply(bcmethod)
    calibration_results[name] = bias_calibration
    validation_results[name] = bias_validation

df_bias_calibration = pd.concat([df.add_prefix(label + '_') for label, df in calibration_results.items()], axis=1)
df_bias_validation = pd.concat([df.add_prefix(label + '_') for label, df in validation_results.items()], axis=1)

def plotbiases(df, title):
    figure, axes = plt.subplots(nrows=2, ncols=1, layout='constrained')

    df_p = df.filter(like='_p')

    minvalue = df_p.min(axis=None)
    maxvalue = df_p.max(axis=None)
    limitvalue = max(abs(minvalue), abs(maxvalue)) * 1.2
    df_p.plot.bar(ax=axes[0], title='Precipitation bias', ylim=(-limitvalue, limitvalue))
    
    df_t = df.filter(like='_t')

    minvalue = df_t.min(axis=None)
    maxvalue = df_t.max(axis=None)
    limitvalue = max(abs(minvalue), abs(maxvalue)) * 1.2
    df_t.plot.bar(ax=axes[1], title='Temperature bias', ylim=(-limitvalue, limitvalue))

    figure.suptitle(title)

plotbiases(df_bias_calibration, title='Bias over calibration period 1976-1995')
plotbiases(df_bias_validation, title='Bias over validation period 1996-2005')

plt.show()