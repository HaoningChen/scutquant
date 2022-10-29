from qlib.data.dataset.handler import DataHandlerLP
import pandas as pd
from qlib.model.base import Model
from typing import Text, Union
from qlib.data.dataset import DatasetH, Dataset
import xgboost
from sklearn.linear_model import LinearRegression

def cov(x, y):
    x_bar = x.mean()
    y_bar = y.mean()
    cov_xy = 0
    for i in range(0, len(x)):
        cov_xy += (x[i] - x_bar) * (y[i] - y_bar)
    cov_xy = cov_xy / len(x)
    return cov_xy


def pearson_corr(x, y):
    x_std = x.std()
    y_std = y.std()
    cov_xy = cov(x, y)
    corr = cov_xy / (x_std * y_std)
    return corr

def dropz(df):
    idx = []
    for c in df.columns:
        if df[c].values.sum() == 0:
            idx.append(c)
    print(str(len(idx))+'columns will be dropped')
    return df.drop(idx, axis=1)

class Hybrid(Model):
    def __init__(self, est=800, colsample_bytree=0.9325, subsample=0.8789, eta=0.0421, reg_alpha=205.6999,
                 reg_lambda=580.9768, max_depth=8, early_stopping_rounds=50,
                 x_train=None, y_train=None, x_valid=None, y_valid=None, x_test=None, y_test=None):

        super(Hybrid, self).__init__()
        self.est = est
        self.colsample = colsample_bytree
        self.subsample = subsample
        self.eta = eta
        self.alpha = reg_alpha
        self.l2 = reg_lambda
        self.max_depth = max_depth
        self.early_stopping_rounds = early_stopping_rounds
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        self.lrg = None
        self.xgb = None

    def fit(self, dataset: Dataset):
        if self.x_train is None:
            dtrain, dvalid = dataset.prepare(
                ["train", "valid"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            dtrain = dtrain.fillna(method='ffill').fillna(0)
            dvalid = dvalid.fillna(method='ffill').fillna(0)

            self.x_train, self.y_train = dtrain["feature"], dtrain["label"]
            self.x_train = dropz(self.x_train)

            self.x_valid, self.y_valid = dvalid["feature"], dvalid["label"]
            self.x_valid = dropz(self.x_valid)

        self.lrg = LinearRegression()
        self.lrg.fit(self.x_train, self.y_train)
        self.xgb = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=self.est, eta=self.eta,
                                        colsample_bytree=self.colsample, subsample=self.subsample,
                                        reg_alpha=self.alpha, reg_lambda=self.l2, max_depth=self.max_depth)
        self.xgb.fit(self.x_train, self.y_train, eval_set=[(self.x_valid, self.y_valid)],
                     early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, dataset: Dataset, segment: Union[Text, slice] = "test") -> object:
        if self.x_test is None:
            dtest = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
            self.x_test, self.y_test = dtest['feature'], dtest['label']

            self.x_test = self.x_test.fillna(method='ffill').fillna(0)
            self.y_test = self.y_test.fillna(method='ffill').fillna(0)
            # print(self.y_test.describe())
            self.x_test = dropz(self.x_test)
        index = self.x_test.index
        lrg_pred = self.lrg.predict(self.x_test)*0.3
        xgb_pred = self.xgb.predict(self.x_test)*0.7
        pred_values = []
        for i in range(0, len(self.x_test.values)):
            pred_values.append(lrg_pred[i]+xgb_pred[i])
        pred = pd.DataFrame(pred_values, columns=['score'], index=index, dtype=float).fillna(method='ffill').fillna(0)
        # print(pred.info())
        print('cov:', cov(pred.values, self.y_test.values) ,'\n', "pearson_corr:", pearson_corr(pred.values, self.y_test.values))
        return pred