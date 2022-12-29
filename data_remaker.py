import pandas as pd
import numpy as np

def feature_engineering(data):
    data.dropna(inplace=True)
    data.drop('CLIENT_ID', axis=1, inplace=True)
    for i in data.columns:
        try:
            data[i] = data[i].astype(int)
        except ValueError:
            continue

    for i in data.columns:
        try:
            data[i] = pd.to_datetime(data[i], format='%Y%m%d')
            data[i] = data[i].apply(lambda x: x.toordinal())
        except ValueError:
            continue
    for i in data.columns:
        try:
            data[i] = data[i].astype(float)
        except ValueError:
            continue
    def convert(s):
        return float(''.join(s.split(',')))

    data['rest_debet_card'] = np.array(list(map(convert, data['rest_debet_card'])))
    data['rest_safe'] = np.array(list(map(convert, data['rest_safe'])))
    data['rest_csf_start'] = np.array(list(map(convert, data['rest_csf_start'])))

    return data