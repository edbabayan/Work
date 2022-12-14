from datetime import datetime
import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype

def featue_engineering (data):
    data.dropna(inplace=True)
    data.drop('CLIENT_ID', axis=1, inplace = True)
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

    return data