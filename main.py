import calendar
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import dataloader
import models

import SBCK
import MLBCK

## DATA

df_calibration = dataloader.load_data(year_start=1976, year_end=1995)
df_validation = dataloader.load_data(year_start=1996, year_end=2005)

x_calibration = df_calibration[['value_pred_p', 'value_pred_t']].values
y_calibration = df_calibration[['value_obsv_p', 'value_obsv_t']].values
x_validation = df_validation[['value_pred_p', 'value_pred_t']].values
y_validation = df_validation[['value_obsv_p', 'value_obsv_t']].values

## CALCULATE ORIGINAL BIAS

df_calibration[['bias_original_p', 'bias_original_t']] = (x_calibration - y_calibration)
df_validation[['bias_original_p', 'bias_original_t']] = (x_validation - y_validation)

## BIAS CORRECTION METHODS

bcmethods = [
    SBCK.CDFt(), # 1 - Statistical quantile-based bias correction
    SBCK.R2D2(), # 2 - Multivariate rank resampling
    MLBCK.N3BC(neuralnetwork=models.ConvLSTMNeuralMVBC()), # 3 - Neural net: Conv + LSTM
    #MLBCK.N3BC(neuralnetwork=models.TransformerBiasCorrection()), # 4 - Neural net: Transformer
    #MLBCK.N3BC(neuralnetwork=models.FCNeuralMVBC()), # 5 - Fully connected naive neural net
    #MLBCK.N3BC(neuralnetwork=models.ConvNeuralMVBC()), # 6 - Convolutional naive neural net
    MLBCK.N3BC(neuralnetwork=models.LSTMNeuralMVBC()),  # 7 - LSTM
]

for bcmethod in bcmethods:

    name = bcmethod.name if hasattr(bcmethod, 'name') else bcmethod.__class__.__name__

    if (len(inspect.signature(bcmethod.fit).parameters) == 3):
        # fit bcmethod with calibration data
        bcmethod.fit(y_calibration, x_calibration, x_validation)
        # correct bias of validation and calibration data using bcmethod
        z_validation, z_calibration = bcmethod.predict(x_validation, x_calibration)
    else:
        # fit bcmethod with calibration data
        bcmethod.fit(y_calibration, x_calibration)
        # correct bias of calibration data using bcmethod
        z_calibration = bcmethod.predict(x_calibration)
        # correct bias of validation data using bcmethod
        z_validation = bcmethod.predict(x_validation)

    df_calibration[[f"bias_{name}_p", f"bias_{name}_t"]] = (z_calibration - y_calibration)
    df_validation[[f"bias_{name}_p", f"bias_{name}_t"]] = (z_validation - y_validation)

## PLOT RESULTS

df_validation_bias_p = df_validation.filter(like='bias').filter(like='_p').groupby(df_validation.index.month).mean()
df_validation_bias_t = df_validation.filter(like='bias').filter(like='_t').groupby(df_validation.index.month).mean()

minvalue = min(df_validation_bias_p.min().min(), df_validation_bias_t.min().min())
maxvalue = max(df_validation_bias_p.max().max(), df_validation_bias_t.max().max())
limitvalue = max(abs(minvalue), abs(maxvalue)) * 1.2

figure, axes = plt.subplots(nrows=2, ncols=1, layout='constrained')

df_validation_bias_p.plot.bar(ax=axes[0], title='Precipitation bias', xlabel='Month', ylabel='Bias', ylim=(-limitvalue, limitvalue))

df_validation_bias_t.plot.bar(ax=axes[1], title='Temperature bias', xlabel='Month', ylabel='Bias', ylim=(-limitvalue, limitvalue))

figure.suptitle('Average bias per month over validation period 1996-2005')

print('mean precipitation bias')
print(df_validation_bias_p.abs().mean())

print('mean temperature bias')
print(df_validation_bias_t.abs().mean())

plt.show()