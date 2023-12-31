import torch
from torch import Tensor
import torch.nn.functional as f
import numpy as np
from pandas import DataFrame, Series, concat

"""
目前models模块使用pytorch实现, 这样可以更好地接入图神经网络模块

输入仍然可以是有多重索引 [(datetime, instrument)] 的DataFrame和Series, 但模型训练之前会自动将数据按天拆成一个list, 并以一天的数据量作为batch
(所以batch size是会变的)

增加了MSEPlus函数, 可以通过调参精确控制预测值与某个变量的相关系数大小
"""


def get_daily_inter(data: Series | DataFrame, shuffle=False):
    daily_count = data.groupby(level=0).size().values
    daily_index = np.roll(np.cumsum(daily_count), 1)
    daily_index[0] = 0
    if shuffle:
        daily_shuffle = list(zip(daily_index, daily_count))
        np.random.shuffle(daily_shuffle)
        daily_index, daily_count = zip(*daily_shuffle)
    return daily_index, daily_count


def from_pandas_to_list(x, for_cnn: bool = False):
    if isinstance(x, DataFrame) or isinstance(x, Series):
        dataset = []
        daily_index, daily_count = get_daily_inter(x)
        for index, count in zip(daily_index, daily_count):
            batch = slice(index, index + count)
            data_slice = x.iloc[batch]
            value = data_slice.values.reshape(-1, data_slice.shape[1], 1) if for_cnn else data_slice.values
            if value.ndim == 1:
                dataset.append(torch.from_numpy(np.squeeze(value)).to(torch.float32).view(-1, 1))
            else:
                dataset.append(torch.from_numpy(value).to(torch.float32))
        return dataset
    else:
        return x


def transform_data(x_train, y_train, x_valid, y_valid, z_train=None, z_valid=None, for_cnn: bool = False):
    """
    将DataFrame或Series拆成list, 并根据模型类型对数据的shape进行调整
    """
    x_train, x_valid = from_pandas_to_list(x_train, for_cnn), from_pandas_to_list(x_valid, for_cnn)
    y_train, y_valid = from_pandas_to_list(y_train), from_pandas_to_list(y_valid)
    if z_train is not None:
        z_train = from_pandas_to_list(z_train)
    if z_valid is not None:
        z_valid = from_pandas_to_list(z_valid)
    return x_train, y_train, x_valid, y_valid, z_train, z_valid


def calc_tensor_corr(x: Tensor, y: Tensor):
    if x.shape != y.shape:
        raise ValueError("The shapes of x and y must be the same.")
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    std_x = torch.std(x)
    std_y = torch.std(y)
    return torch.mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)


def split_dataset_by_index(dataset: list, train_index, test_index):
    """
    用于滚动训练
    """
    f_array = np.zeros(shape=(len(dataset),)).astype(bool)
    train_mask, test_mask = f_array.copy(), f_array.copy()
    train_mask[train_index] = True
    test_mask[test_index] = True
    d_train = [d for d, mask in zip(dataset, train_mask.tolist()) if mask]
    d_test = [d for d, mask in zip(dataset, test_mask.tolist()) if mask]
    return d_train, d_test


class MSEPlus(torch.nn.Module):
    def __init__(self):
        """
        当我们想控制预测值和某个变量的相关系数时

        self.loss = MSEPlus()
        self.loss((x, y, size))
        """
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, inputs: tuple):
        x, y, z = inputs[0], inputs[1], inputs[2]
        mse_loss = self.mse(x, y)
        ic_loss = calc_tensor_corr(x, z)
        # return mse_loss * (1 + 0.1 * (torch.abs(ic_loss) - ic_loss))  # 对ic为负的双倍惩罚, 对ic为正的不作额外惩罚
        return mse_loss * (1 + 0.05 * torch.abs(ic_loss))


class MLP(torch.nn.Module):
    def __init__(self, input_shape: int, hidden_shape: int, output_shape: int, epochs: int = 10, loss: str = "mse_loss",
                 lr: float = 1e-3, weight_decay: float = 5e-4, dropout: float = 0.3, model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.feature_filter = torch.nn.Linear(in_features=self.input_shape, out_features=self.hidden_shape, bias=False)
        self.hidden_layer = torch.nn.Linear(in_features=self.hidden_shape, out_features=self.hidden_shape)
        self.out_layer = torch.nn.Linear(in_features=self.hidden_shape, out_features=self.output_shape)
        self.jump_layer = torch.nn.Linear(in_features=self.input_shape, out_features=self.hidden_shape)
        self.bn = torch.nn.BatchNorm1d(self.hidden_shape)
        self.epochs = epochs
        self.loss = loss if loss != "special" else MSEPlus()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.model = model
        self.optimizer = None

    def init_model(self):
        self.model = MLP(input_shape=self.input_shape, hidden_shape=self.hidden_shape,
                         output_shape=self.output_shape, epochs=self.epochs, loss=self.loss, lr=self.lr,
                         weight_decay=self.weight_decay, dropout=self.dropout).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x):
        x1 = f.relu(self.jump_layer(x))
        x1 = f.dropout(x1, p=self.dropout, training=self.training)

        x = self.feature_filter(x)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = self.hidden_layer(x)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)

        return self.out_layer(self.bn(x + x1))

    def get_loss(self, x, y, z=None):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x)
        if isinstance(self.loss, str):
            loss = eval("f." + self.loss + "(out, y)")
        else:
            loss = self.loss((out, y, z))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def predict_(self, x):
        self.model.eval()
        pred = self.model(x)
        return pred

    @torch.no_grad()
    def test(self, x, y, z=None):
        pred_ = self.predict_(x)
        if isinstance(self.loss, str):
            loss = eval("f." + self.loss + "(pred_, y)")
        else:
            loss = self.loss((pred_, y, z))
        return float(loss)

    def fit(self, x_train, y_train, x_valid, y_valid, z_train=None, z_valid=None):
        if self.model is None:
            self.init_model()

        x_train, y_train, x_valid, y_valid, z_train, z_valid = transform_data(x_train, y_train, x_valid, y_valid,
                                                                              z_train, z_valid)

        for epoch in range(1, self.epochs + 1):
            total_loss_train = 0
            total_loss_val = 0
            val_ic = 0
            for i in range(len(x_train)):
                loss_train = self.get_loss(x=x_train[i], y=y_train[i], z=z_train[i] if z_train is not None else None)
                total_loss_train += loss_train
            for i in range(len(x_valid)):
                loss_val = self.test(x=x_valid[i], y=y_valid[i], z=z_valid[i] if z_valid is not None else None)
                total_loss_val += loss_val
                val_ic += float(calc_tensor_corr(self.predict_(x_valid[i]), y_valid[i]))
            print("Epoch:", epoch, "loss:", total_loss_train / len(x_train), "val_loss:",
                  total_loss_val / len(x_valid), "val_ic:", val_ic / len(x_valid))

    def predict_pandas(self, x: DataFrame, for_cnn: bool = False) -> Series:
        index = x.index
        x = from_pandas_to_list(x, for_cnn)
        result = []
        for batch in x:
            result.append(Series(self.predict_(batch).view(-1, )))
        series = concat(result, axis=0)
        series.index = index
        return series

    def save(self, path: str = "model.pth"):
        torch.save(self.state_dict(), path)
