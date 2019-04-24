import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

class CustomScaler(TransformerMixin):
    def __init__(self, continuous_idx, dummies_idx):
        self.continuous_idx = continuous_idx
        self.dummies_idx = dummies_idx
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[:, self.continuous_idx])
        return self

    def transform(self, X, y=None, copy=None):
        X_head = self.scaler.transform(X[:, self.continuous_idx])
        return np.concatenate((X_head, X[:, self.dummies_idx]), axis=1)

def BalanceDataset(train_data):
    target = ['Response']
    col = train_data.columns[~train_data.columns.isin(target)]

    train_label = train_data[target]
    train_data = train_data[col]

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(train_data, train_label)

    df_over = pd.DataFrame(X_resampled, columns=col)
    df_over['Response'] = y_resampled
    return df_over
