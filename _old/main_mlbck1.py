import calendar
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import SBCK
# import MLBCK

obsv_p = pd.read_csv('./data/observations_precipitation.csv', index_col=['year', 'month', 'day'])
obsv_t = pd.read_csv('./data/observations_temperature.csv', index_col=['year', 'month', 'day'])
pred_p = pd.read_csv('./data/predictions_precipitation.csv', index_col=['year', 'month', 'day'])
pred_t = pd.read_csv('./data/predictions_temperature.csv', index_col=['year', 'month', 'day'])

precipitation = pd.merge(obsv_p, pred_p, on=['year', 'month', 'day'], suffixes=("_obsv", "_pred"))
temperature = pd.merge(obsv_t, pred_t, on=['year', 'month', 'day'], suffixes=("_obsv", "_pred"))

df = pd.merge(precipitation, temperature, on=['year', 'month', 'day'], suffixes=("_p", "_t"))

df = df.set_index(pd.to_datetime(df.index.to_frame()))

# calendarposition = df.index.day_of_year / (365 + df.index.is_leap_year * 1)
# df['calendarposition_x'] = (np.cos(calendarposition * np.pi * 2) + 1) / 2
# df['calendarposition_y'] = (np.sin(calendarposition * np.pi * 2) + 1) / 2

# df['season'] = (df.index.month // 3) % 4

df_calibration = df[(df.index.year >= 1976) & (df.index.year <= 1995)]
df_validation = df[(df.index.year >= 1996) & (df.index.year <= 2005)]

Y_calibration = df_calibration[['value_obsv_p', 'value_obsv_t']].values
X_calibration = df_calibration[['value_pred_p', 'value_pred_t']].values
Y_validation = df_validation[['value_obsv_p', 'value_obsv_t']].values
X_validation = df_validation[['value_pred_p', 'value_pred_t']].values

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
    df_bias_calibration = calculatebias(Z_calibration, Y_calibration, df_calibration.index)

    # calculate bias of corrected validation data
    df_bias_validation = calculatebias(Z_validation, Y_validation, df_validation.index)

    return df_bias_calibration, df_bias_validation

bcmethods = [
    # MLBCK.N3BC(),
    # MLBCK.ConvBC()
]
bcmethods = {bcmethod.__class__.__name__: bcmethod for bcmethod in bcmethods}

calibration_results = { 'original': calculatebias(X_calibration, Y_calibration, df_calibration.index) }
validation_results = { 'original': calculatebias(X_validation, Y_validation, df_validation.index) }

for name, bcmethod in bcmethods.items():
    bias_calibration, bias_validation = apply(bcmethod)
    calibration_results[name] = bias_calibration
    validation_results[name] = bias_validation
    print(bias_validation)

df_bias_calibration = pd.concat([df.add_prefix(label + '_') for label, df in calibration_results.items()], axis=1)
df_bias_validation = pd.concat([df.add_prefix(label + '_') for label, df in validation_results.items()], axis=1)

def plotbiases(df, title):
    df = df.set_index(df.index.map(lambda i: calendar.month_name[i]))

    figure, axes = plt.subplots(nrows=2, ncols=1, layout='constrained')

    df_p = df.filter(like='_p')

    minvalue = df_p.min(axis=None)
    maxvalue = df_p.max(axis=None)
    limitvalue = max(abs(minvalue), abs(maxvalue)) * 1.2
    df_p.plot.bar(ax=axes[0], title='Precipitation bias', xlabel='Month', ylabel='Bias', ylim=(-limitvalue, limitvalue))
    
    df_t = df.filter(like='_t')

    minvalue = df_t.min(axis=None)
    maxvalue = df_t.max(axis=None)
    limitvalue = max(abs(minvalue), abs(maxvalue)) * 1.2
    df_t.plot.bar(ax=axes[1], title='Temperature bias', ylim=(-limitvalue, limitvalue))

    figure.suptitle(title)

# plotbiases(df_bias_calibration, title='Bias over calibration period 1976-1995')
# plotbiases(df_bias_validation, title='Bias over validation period 1996-2005')

# plt.show()