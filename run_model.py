import os
import torch
import sklearn.model_selection
import numpy as np
import matplotlib.pyplot as plt

import dataloader
import models

seed = 9000

## DATA
os.makedirs("checkpoints", exist_ok=True)

df_calibration = dataloader.load_data(year_start=1976, year_end=1995)
df_validation = dataloader.load_data(year_start=1996, year_end=2005)

x_calibration = df_calibration[['value_pred_p', 'value_pred_t']].values.reshape(-1, 365, 2)
y_calibration = df_calibration[['value_obsv_p', 'value_obsv_t']].values.reshape(-1, 365, 2)
x_validation = df_validation[['value_pred_p', 'value_pred_t']].values.reshape(-1, 365, 2)
y_validation = df_validation[['value_obsv_p', 'value_obsv_t']].values.reshape(-1, 365, 2)

## NORMALIZATION

# calculate min-max values based on calibration data
min_values = x_calibration.min(axis=(0, 1))
max_values = x_calibration.max(axis=(0, 1))
# min-max normalization of all predictions
x_calibration = (x_calibration - min_values) / (max_values - min_values)
x_validation = (x_validation - min_values) / (max_values - min_values)
# min-max normalization of all observations
y_calibration = (y_calibration - min_values) / (max_values - min_values)
y_validation = (y_validation - min_values) / (max_values - min_values)

## SETUP

model = models.ConvNeuralMVBC()
#model = models.LSTMNeuralMVBC()
#model = models.TransformerBiasCorrection() #Long term
#model = models.ConvLSTMNeuralMVBC()


lossfunction = torch.nn.MSELoss()
#lossfunction = nn.L1Loss()  # Using MAE
# lossfunction = nn.SmoothL1Loss(beta=1.0)  # Using Smooth L1 Loss
# lossfunction = nn.HuberLoss(delta=1.0)  # Using Huber Loss

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Using SGD
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)  # Using RMSprop
# optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Using AdamW
## TRAINING

x_train = x_calibration[:, :, :]
x_test = x_validation[:, :, :]
y_train = y_calibration[:, :, :]
y_test = y_validation[:, :, :]

train_losses = []
test_losses = []

for epoch in range(1000):

    model.train()
    optimizer.zero_grad()

    # sample a random batch
    idx = np.random.choice(len(x_train), len(x_test))
    x_train_sample = torch.tensor(x_train[idx, :, :], dtype=torch.float32)
    y_train_sample = torch.tensor(y_train[idx, :, :], dtype=torch.float32)

    z_train_sample = model(x_train_sample)

    train_loss = lossfunction(z_train_sample, y_train_sample)

    train_loss.backward()

    optimizer.step()

    if (epoch % 10 == 0):

        model.eval()

        x_test_sample = torch.tensor(x_test, dtype=torch.float32)
        y_test_sample = torch.tensor(y_test, dtype=torch.float32)

        z_test_sample = model(x_test_sample)

        test_loss = lossfunction(z_test_sample, y_test_sample)

        print(f"epoch: {epoch} - test loss: {test_loss}")

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if test_loss <= min(test_losses):
            torch.save(model.state_dict(), os.path.join('checkpoints', model.__class__.__name__))

min_test_loss = min(test_losses)
print(f"minimum test loss: {min_test_loss}")

plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

## EVALUATION

model.load_state_dict(torch.load(os.path.join('checkpoints', model.__class__.__name__), weights_only=True))
model.eval()

x_validation_sample = torch.tensor(x_validation, dtype=torch.float32)

z_validation_sample = model(x_validation_sample)

z_validation = z_validation_sample.detach().numpy()

# min-max de-normalization
x_validation = min_values + x_validation * (max_values - min_values)
y_validation = min_values + y_validation * (max_values - min_values)
z_validation = min_values + z_validation * (max_values - min_values)

df_validation[['bias_original_p', 'bias_original_t']] = (x_validation - y_validation).reshape(-1, 2)
df_validation[['bias_corrected_p', 'bias_corrected_t']] = (z_validation - y_validation).reshape(-1, 2)

## PLOT RESULTS

df_validation_bias_p = df_validation[['bias_original_p', 'bias_corrected_p']].groupby(df_validation.index.month).mean()
df_validation_bias_t = df_validation[['bias_original_t', 'bias_corrected_t']].groupby(df_validation.index.month).mean()

minvalue = min(df_validation_bias_p.min().min(), df_validation_bias_t.min().min())
maxvalue = max(df_validation_bias_p.max().max(), df_validation_bias_t.max().max())
limitvalue = max(abs(minvalue), abs(maxvalue)) * 1.2

figure, axes = plt.subplots(nrows=2, ncols=1, layout='constrained')

df_validation_bias_p.plot.bar(ax=axes[0], title='Precipitation bias', xlabel='Month', ylabel='Bias', ylim=(-limitvalue, limitvalue))

df_validation_bias_t.plot.bar(ax=axes[1], title='Temperature bias', xlabel='Month', ylabel='Bias', ylim=(-limitvalue, limitvalue))

figure.suptitle('Average bias per month over validation period 1996-2005')

plt.show()

## PLOT DATA SHIFT

# df_validation[['bias_original_p', 'bias_corrected_p']]
