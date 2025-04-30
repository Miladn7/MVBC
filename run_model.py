import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import dataloader
import models

seed = 9000
window = 5  # 5-day sliding window

## DATA
os.makedirs("checkpoints", exist_ok=True)

df_calibration = dataloader.load_data(year_start=1976, year_end=1995)
df_validation = dataloader.load_data(year_start=1996, year_end=2005)


# 🔥 HEATWAVE & DROUGHT DETECTION
def detect_heatwaves(df, threshold_quantile=0.95, min_duration=5):
    temp = df['value_obsv_t']
    threshold = temp.quantile(threshold_quantile)
    hot_days = temp >= threshold
    heatwave_flags = pd.Series(False, index=df.index)
    streak = 0
    for i in range(len(hot_days)):
        if hot_days.iloc[i]:
            streak += 1
            if streak >= min_duration:
                heatwave_flags.iloc[i - min_duration + 1: i + 1] = True
        else:
            streak = 0
    return heatwave_flags


def detect_droughts(df, threshold_quantile=0.05, min_duration=5):
    precip = df['value_obsv_p']
    threshold = precip.quantile(threshold_quantile)
    dry_days = precip < threshold
    drought_flags = pd.Series(False, index=df.index)
    streak = 0
    for i in range(len(dry_days)):
        if dry_days.iloc[i]:
            streak += 1
            if streak >= min_duration:
                drought_flags.iloc[i - min_duration + 1 : i + 1] = True
        else:
            streak = 0
    return drought_flags



# 🔁 BUILD 5-DAY SEQUENCES + EXTREME EVENT WEIGHTING (compound-aware)
def build_sequences_with_flags(df, window=5):
    x_seq, y_seq, weights = [], [], []

    # Detect individual events
    heatwave_flags = detect_heatwaves(df)
    drought_flags = detect_droughts(df)

    # Convert to integers
    heat_flags = heatwave_flags.astype(int)
    drought_flags = drought_flags.astype(int)

    # Combined score: 0 (none), 1 (one of them), 2 (both)
    compound_flags = heat_flags + drought_flags

    for year, group in df.groupby(df.index.year):
        if len(group) < window:
            continue

        x = group[['value_pred_p', 'value_pred_t']].values
        y = group[['value_obsv_p', 'value_obsv_t']].values

        hf = heat_flags[group.index]
        df_ = drought_flags[group.index]
        cf = compound_flags[group.index]

        for i in range(len(group) - window + 1):
            x_seq.append(x[i:i + window])
            y_seq.append(y[i:i + window])

            # Check flags in this 5-day window
            h = hf.iloc[i:i + window].any()
            d = df_.iloc[i:i + window].any()
            c = cf.iloc[i:i + window].sum()

            if h and d:
                w = 2.5  # compound event
            elif h or d:
                w = 2.0  # one extreme
            else:
                w = 1.0  # normal

            # Optional: Boost for very intense compound windows
            if c >= window:
                w = 3.0  # every day is extreme

            weights.append(w)

    return np.array(x_seq), np.array(y_seq), np.array(weights)


# ✅ LOAD AND PROCESS SEQUENCES
x_calibration, y_calibration, weights_cal = build_sequences_with_flags(df_calibration, window)
x_validation, y_validation, weights_val = build_sequences_with_flags(df_validation, window)

# ✅ NORMALIZATION
min_values = x_calibration.min(axis=(0, 1))
max_values = x_calibration.max(axis=(0, 1))

x_calibration = (x_calibration - min_values) / (max_values - min_values)
x_validation = (x_validation - min_values) / (max_values - min_values)
y_calibration = (y_calibration - min_values) / (max_values - min_values)
y_validation = (y_validation - min_values) / (max_values - min_values)

# ✅ MODEL SETUP
model = models.ConvNeuralMVBC()
#model = models.AdvancedLSTMNeuralMVBC()
# model = models.LSTMNeuralMVBC()
# model = models.ConvLSTMNeuralMVBC()


# model = models.LSTMNeuralMVBC()
# model = models.ConvLSTMNeuralMVBC()

class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, weights):
        loss = ((pred - target) ** 2).mean(dim=(1, 2))  # per sequence
        weighted = weights * loss
        if self.reduction == 'sum':
            return torch.sum(weighted)
        elif self.reduction == 'none':
            return weighted
        else:
            return torch.mean(weighted)


# lossfunction = torch.nn.MSELoss()
# lossfunction = nn.MSELoss()
# lossfunction = nn.L1Loss()  # Using MAE
# lossfunction = nn.SmoothL1Loss(beta=1.0)  # Using Smooth L1 Loss
# lossfunction = nn.HuberLoss(delta=1.0)  # Using Huber Loss
lossfunction = WeightedMSELoss()

# optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Using AdamW
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# ✅ TRAINING
x_train, x_test = x_calibration, x_validation
y_train, y_test = y_calibration, y_validation
weights_cal = np.array(weights_cal)
weights_val = np.array(weights_val)

train_losses = []
test_losses = []

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()

    # sample a random batch

    idx = np.random.choice(len(x_train), len(x_test))
    x_train_sample = torch.tensor(x_train[idx], dtype=torch.float32)
    y_train_sample = torch.tensor(y_train[idx], dtype=torch.float32)
    w_train_sample = torch.tensor(weights_cal[idx], dtype=torch.float32)

    z_train_sample = model(x_train_sample)

    if z_train_sample.shape[1] != y_train_sample.shape[1]:
        z_train_sample = z_train_sample[:, :y_train_sample.shape[1], :]

    train_loss = lossfunction(z_train_sample, y_train_sample, w_train_sample)
    train_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        x_test_sample = torch.tensor(x_test, dtype=torch.float32)
        y_test_sample = torch.tensor(y_test, dtype=torch.float32)
        w_test_sample = torch.tensor(weights_val, dtype=torch.float32)

        z_test_sample = model(x_test_sample)

        if z_test_sample.shape[1] != y_test_sample.shape[1]:
            z_test_sample = z_test_sample[:, :y_test_sample.shape[1], :]

        test_loss = lossfunction(z_test_sample, y_test_sample, w_test_sample)

        print(f"epoch: {epoch} - test loss: {test_loss.item():.6f}")

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if test_loss <= min(test_losses):
            torch.save(model.state_dict(), os.path.join('checkpoints', model.__class__.__name__))

# ✅ PLOT TRAINING CURVE
plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# ✅ EVALUATION
model.load_state_dict(torch.load(os.path.join('checkpoints', model.__class__.__name__), weights_only=True))
model.eval()

x_val_tensor = torch.tensor(x_validation, dtype=torch.float32)
z_val_tensor = model(x_val_tensor)

if z_val_tensor.shape[1] != y_validation.shape[1]:
    z_val_tensor = z_val_tensor[:, :y_validation.shape[1], :]

z_validation = z_val_tensor.detach().numpy()

# ✅ Min\Max DE-NORMALIZATION
x_validation = min_values + x_validation * (max_values - min_values)
y_validation = min_values + y_validation * (max_values - min_values)
z_validation = min_values + z_validation * (max_values - min_values)

# ✅ MONTHLY BIAS PLOTTING
center_idx = [df_validation.index[i + window // 2] for i in range(len(z_validation))]
df_eval = pd.DataFrame(index=pd.DatetimeIndex(center_idx))
df_eval[['bias_original_p', 'bias_original_t']] = (x_validation - y_validation)[:, window // 2, :]
df_eval[['bias_corrected_p', 'bias_corrected_t']] = (z_validation - y_validation)[:, window // 2, :]

df_validation_bias_p = df_eval[['bias_original_p', 'bias_corrected_p']].groupby(df_eval.index.month).mean()
df_validation_bias_t = df_eval[['bias_original_t', 'bias_corrected_t']].groupby(df_eval.index.month).mean()

minvalue = min(df_validation_bias_p.min().min(), df_validation_bias_t.min().min())
maxvalue = max(df_validation_bias_p.max().max(), df_validation_bias_t.max().max())
limitvalue = max(abs(minvalue), abs(maxvalue)) * 1.2

fig, axes = plt.subplots(nrows=2, ncols=1, layout='constrained')
df_validation_bias_p.plot.bar(ax=axes[0], title='Precipitation bias', xlabel='Month', ylabel='Bias',
                              ylim=(-limitvalue, limitvalue))
df_validation_bias_t.plot.bar(ax=axes[1], title='Temperature bias', xlabel='Month', ylabel='Bias',
                              ylim=(-limitvalue, limitvalue))

fig.suptitle('Average bias per month (5-day windows) – Validation Period 1996–2005')

plt.show()

## PLOT DATA SHIFT

# df_validation[['bias_original_p', 'bias_corrected_p']]