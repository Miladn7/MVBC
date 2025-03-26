import pandas as pd

def load_data(year_start, year_end):
    pred_p = pd.read_csv('./data/predictions_precipitation.csv', index_col=['year', 'month', 'day'])
    pred_t = pd.read_csv('./data/predictions_temperature.csv', index_col=['year', 'month', 'day'])
    obsv_p = pd.read_csv('./data/observations_precipitation.csv', index_col=['year', 'month', 'day'])
    obsv_t = pd.read_csv('./data/observations_temperature.csv', index_col=['year', 'month', 'day'])

    precipitation = pd.merge(pred_p, obsv_p, on=['year', 'month', 'day'], suffixes=("_pred", "_obsv"))
    temperature = pd.merge(pred_t, obsv_t, on=['year', 'month', 'day'], suffixes=("_pred", "_obsv"))

    df = pd.merge(precipitation, temperature, on=['year', 'month', 'day'], suffixes=("_p", "_t"))

    df = df.set_index(pd.to_datetime(df.index.to_frame()))

    df = df.drop(df[(df.index.month == 2) & (df.index.day == 29)].index) # drop february 29th for leap years

    df = df[(df.index.year >= year_start) & (df.index.year <= year_end)] # select years between start and end

    return df
