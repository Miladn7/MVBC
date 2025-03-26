import os
import torch
import sklearn.model_selection
import numpy as np
import matplotlib.pyplot as plt

import dataloader
import models

seed = 9000

## DATA

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

model = models.FCNeuralMVBC()

lossfunction = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

## TRAINING

kfold = sklearn.model_selection.KFold(n_splits=5, shuffle=True)

fold_losses = []

for fold, (train_ids, test_ids) in enumerate(kfold.split(x_calibration)):

    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    x_train = x_calibration[train_ids, :, :]
    x_test = x_calibration[test_ids, :, :]
    y_train = y_calibration[train_ids, :, :]
    y_test = y_calibration[test_ids, :, :]

    train_losses = []
    test_losses = []

    for epoch in range(1000):

        model.train()
            
        model.zero_grad()

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

            # print(f"fold: {fold} - epoch: {epoch} - test loss: {test_loss}")

            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())

            if test_loss <= min(test_losses):
                torch.save(model.state_dict(), os.path.join('checkpoints', model.__class__.__name__ + str(fold)))

    min_test_loss = min(test_losses)
    print(f"fold: {fold} - minimum test loss: {min_test_loss}")

    fold_losses.append(min_test_loss)

    if min_test_loss <= min(fold_losses):
        modelparameters = torch.load(os.path.join('checkpoints', model.__class__.__name__ + str(fold)), weights_only=True)
        torch.save(modelparameters, os.path.join('checkpoints', model.__class__.__name__))

    # plt.plot(train_losses, label='training loss')
    # plt.plot(test_losses, label='test loss')
    # plt.legend()
    # plt.show()

average_min_test_loss = sum(fold_losses) / len(fold_losses)
print(f"average minimum test loss: {average_min_test_loss}")

overall_min_test_loss_fold, overall_min_test_loss = min(enumerate(fold_losses), key=lambda l: l[1])
print(f"overall minimum test loss: {overall_min_test_loss} (fold: {overall_min_test_loss_fold})")

## EVALUATION

# find model version with best result on validation data

eval_losses = []

for fold in range(kfold.get_n_splits()):

    model.load_state_dict(torch.load(os.path.join('checkpoints', model.__class__.__name__ + str(fold)), weights_only=True))
    model.eval()

    x_validation_sample = torch.tensor(x_validation, dtype=torch.float32)
    y_validation_sample = torch.tensor(y_validation, dtype=torch.float32)

    z_validation_sample = model(x_validation_sample)

    eval_loss = lossfunction(z_validation_sample, y_validation_sample)

    print(f"fold: {fold} - evaluation loss: {eval_loss}")

    eval_losses.append(eval_loss.item())

    if eval_loss <= min(eval_losses):
        torch.save(model.state_dict(), os.path.join('checkpoints', model.__class__.__name__))

average_eval_loss = sum(eval_losses) / len(eval_losses)
print(f"average evaluation loss: {average_eval_loss}")

min_eval_loss_fold, min_eval_loss = min(enumerate(eval_losses), key=lambda l: l[1])
print(f"minimum evaluation loss: {min_eval_loss} (fold: {min_eval_loss_fold})")

# calculate bias for best model version on validation data

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
