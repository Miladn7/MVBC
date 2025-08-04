import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calendar
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
    x_seq, y_seq, weights, labels = [], [], [], []

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
        dates = group.index

        for i in range(len(group) - window + 1):
            x_seq.append(x[i:i + window])
            y_seq.append(y[i:i + window])

            # Check flags in this 5-day window
            h = hf.iloc[i:i + window].any()
            d = df_.iloc[i:i + window].any()
            c = cf.iloc[i:i + window].sum()

            if h and d:
                w = 2.5  # compound event
                label = "compound"
            elif h or d:
                w = 2.0  # one extreme
                label = "single_extreme"
            else:
                w = 1.0  # normal
                label = "normal"


            # Optional: Boost for very intense compound windows
            if c >= window:
                w = 3.0  # every day is extreme
                label = "full_extreme"

            weights.append(w)
            labels.append(label)


    return np.array(x_seq), np.array(y_seq), np.array(weights), np.array(labels)


# ✅ LOAD AND PROCESS SEQUENCES
x_calibration, y_calibration, weights_cal, labels_cal = build_sequences_with_flags(df_calibration, window)
x_validation, y_validation, weights_val, labels_val = build_sequences_with_flags(df_validation, window)

# ✅ Preserve original for consistent bias calculation
x_val_orig = x_validation.copy()
y_val_orig = y_validation.copy()

# ✅ NORMALIZATION
# Separate per-variable min-max
min_values = x_calibration.min(axis=(0, 1))  # shape: (2,)
max_values = x_calibration.max(axis=(0, 1))

# Use broadcasting for correct shape (samples, window, 2)
x_calibration = (x_calibration - min_values[None, None, :]) / (max_values - min_values)[None, None, :]
x_validation = (x_validation - min_values[None, None, :]) / (max_values - min_values)[None, None, :]

y_calibration = (y_calibration - min_values[None, None, :]) / (max_values - min_values)[None, None, :]
y_validation = (y_validation - min_values[None, None, :]) / (max_values - min_values)[None, None, :]

# ✅ MODEL SETUP
#model = models.ConvNeuralMVBC()
model = models.AdvancedLSTMNeuralMVBC()
#model = models.LSTMNeuralMVBC()
#model = models.ConvLSTMNeuralMVBC()

class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, weights):
        # Compute per-sample MSE over sequences
        loss = ((pred - target) ** 2).mean(dim=(1, 2))  # shape: (batch,)
        weighted = weights * loss
        if self.reduction == 'sum':
            return torch.sum(weighted)
        elif self.reduction == 'none':
            return weighted
        else:
            return torch.mean(weighted)


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred, target, weights):
        error = pred - target
        is_small_error = torch.abs(error) < self.delta
        squared_loss = 0.5 * error**2
        linear_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        loss = torch.where(is_small_error, squared_loss, linear_loss)
        loss = loss.mean(dim=(1, 2))  # average over sequence
        weighted = weights * loss
        return torch.mean(weighted) if self.reduction == 'mean' else weighted


class VariableWeightedMSELoss(nn.Module):
    def __init__(self, temp_weight=1.0, precip_weight=2.0):  # boost precip
        super().__init__()
        self.temp_weight = temp_weight
        self.precip_weight = precip_weight

    def forward(self, pred, target):
        # Assume shape: (batch, time, features)
        loss_temp = ((pred[:, :, 1] - target[:, :, 1]) ** 2).mean()
        loss_precip = ((pred[:, :, 0] - target[:, :, 0]) ** 2).mean()
        return self.temp_weight * loss_temp + self.precip_weight * loss_precip


#lossfunction = torch.nn.MSELoss()
lossfunction = nn.MSELoss()
#lossfunction = nn.L1Loss()  # Using MAE
#lossfunction = nn.SmoothL1Loss(beta=1.0)  # Using Smooth L1 Loss
#lossfunction = nn.HuberLoss(delta=1.0)  # Using Huber Loss
#lossfunction = WeightedMSELoss()
#lossfunction = VariableWeightedMSELoss()
#lossfunction = WeightedHuberLoss(delta=1.0)


optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Using AdamW
#optimizer = optim.Adam(model.parameters(), lr=0.0005)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# ✅ TRAINING
x_train, x_test = x_calibration, x_validation
y_train, y_test = y_calibration, y_validation
weights_cal = np.array(weights_cal)
weights_val = np.array(weights_val)

train_losses = []
test_losses = []

for epoch in range(1000):batch_size = 128

for epoch in range(1000):
    model.train()
    total_loss = 0.0

    for i in range(0, len(x_train), batch_size):
        x_batch = torch.tensor(x_train[i:i+batch_size], dtype=torch.float32)
        y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)
        w_batch = torch.tensor(weights_cal[i:i+batch_size], dtype=torch.float32)

        optimizer.zero_grad()
        z_batch = model(x_batch)

        if z_batch.shape[1] != y_batch.shape[1]:
            z_batch = z_batch[:, :y_batch.shape[1], :]

        loss = lossfunction(z_batch, y_batch ) #w_batch
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / (len(x_train) // batch_size)

    if epoch % 10 == 0:
        model.eval()
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        w_test_tensor = torch.tensor(weights_val, dtype=torch.float32)

        z_test_tensor = model(x_test_tensor)

        if z_test_tensor.shape[1] != y_test_tensor.shape[1]:
            z_test_tensor = z_test_tensor[:, :y_test_tensor.shape[1], :]

        test_loss = lossfunction(z_test_tensor, y_test_tensor ) #w_test_tensor

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Test Loss: {test_loss.item():.6f}")

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss.item())

        if test_loss.item() <= min(test_losses):
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
df_eval[['bias_original_p', 'bias_original_t']] = (x_val_orig - y_val_orig)[:, window // 2, :]
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

#plt.show()

july_mask = pd.Series(center_idx).dt.month == 7
print("Weights used in July:", weights_val[july_mask])

## PLOT DATA SHIFT

#df_validation[['bias_original_p', 'bias_corrected_p']]

#Plot for compund events
event_mask = pd.Series(labels_val, index=center_idx)  # assuming labels_val already exists

for event_type in ['single_extreme', 'compound', 'full_extreme']:
    df_event = df_eval[event_mask == event_type]

    if len(df_event) == 0:
        print(f"No data for: {event_type}")
        continue

    df_event_bias_p = df_event[['bias_original_p', 'bias_corrected_p']].groupby(df_event.index.month).mean()
    df_event_bias_t = df_event[['bias_original_t', 'bias_corrected_t']].groupby(df_event.index.month).mean()

    minvalue = min(df_event_bias_p.min().min(), df_event_bias_t.min().min())
    maxvalue = max(df_event_bias_p.max().max(), df_event_bias_t.max().max())
    limitvalue = max(abs(minvalue), abs(maxvalue)) * 1.2

    fig, axes = plt.subplots(nrows=2, ncols=1, layout='constrained')

    df_event_bias_p.plot.bar(ax=axes[0], title=f'Precipitation bias – {event_type}', xlabel='Month', ylabel='Bias',
                             ylim=(-limitvalue, limitvalue))
    df_event_bias_t.plot.bar(ax=axes[1], title=f'Temperature bias – {event_type}', xlabel='Month', ylabel='Bias',
                             ylim=(-limitvalue, limitvalue))

    fig.suptitle(f'Average bias per month – {event_type.upper()} Events (Validation: 1996–2005)')

    plt.show()

# Load benchmark bias results from SBCK & ML models
df_benchmark = pd.read_csv("results/bias_results_validation.csv", index_col=0, parse_dates=True)
# === BENCHMARK BAR PLOT: 4 METHODS COMPARED PER MONTH ===

df_validation_bias_p = df_benchmark.filter(like='bias').filter(like='_p').groupby(df_benchmark.index.month).mean()
df_validation_bias_t = df_benchmark.filter(like='bias').filter(like='_t').groupby(df_benchmark.index.month).mean()
# 1. Choose method prefixes you want to compare
methods = ['bias_original', 'bias_AdvancedLSTMNeuralMVBC', 'bias_CDFt', 'bias_R2D2']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['Original', 'ML-Corrected', 'CDFt', 'R2D2']

# 2. Prepare month x-axis
month_names = [calendar.month_abbr[m] for m in range(1, 13)]
bar_width = 0.2
x = np.arange(12)  # 12 months

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), layout='constrained')

# 3. Precipitation Bias
for i, method in enumerate(methods):
    bias_values = df_validation.groupby(df_validation.index.month)[f"{method}_p"].mean()
    axes[0].bar(x + i * bar_width, bias_values.values, width=bar_width, label=labels[i], color=colors[i])
axes[0].set_title("Precipitation Bias (Benchmark)")
axes[0].set_ylabel("Bias")
axes[0].set_xticks(x + bar_width * 1.5)
axes[0].set_xticklabels(month_names)
axes[0].legend()

# 4. Temperature Bias
for i, method in enumerate(methods):
    bias_values = df_validation.groupby(df_validation.index.month)[f"{method}_t"].mean()
    axes[1].bar(x + i * bar_width, bias_values.values, width=bar_width, label=labels[i], color=colors[i])
axes[1].set_title("Temperature Bias (Benchmark)")
axes[1].set_ylabel("Bias")
axes[1].set_xticks(x + bar_width * 1.5)
axes[1].set_xticklabels(month_names)
axes[1].legend()

fig.suptitle("Multivariate Bias Correction Benchmark – Monthly Comparison")

plt.show()