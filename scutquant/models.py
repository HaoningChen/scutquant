import torch
from torch import Tensor
import torch.nn.functional as f
import numpy as np
from pandas import DataFrame, Series, concat
from torch_geometric.nn.conv import SAGEConv, GCNConv, GATConv

"""
目前models模块使用pytorch实现, 这样可以更好地接入图神经网络模块

输入仍然可以是有多重索引 [(datetime, instrument)] 的DataFrame和Series, 但模型训练之前会自动将数据按天拆成一个list(GRU需要特殊的处理), 
并以一天的数据量作为batch(所以batch size是会变的)

增加了style_mse函数, 可以通过调参精确控制预测值与某个变量的相关系数大小
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
            if for_cnn:
                value = data_slice.values.reshape(-1, data_slice.shape[1], 1)  # instrument * feat * 1
                # print(value.shape)
            else:
                value = data_slice.values  # instrument * feat
            if value.ndim == 1:
                dataset.append(torch.from_numpy(np.squeeze(value)).to(torch.float32).view(-1, 1))
            else:
                dataset.append(torch.from_numpy(value).to(torch.float32))
        # print(dataset[-1].shape)
        return dataset
    else:
        return x


def from_pandas_to_rnn(x: DataFrame | Series, fillna: bool = False):
    if isinstance(x, Series):
        n_feat = 1
    else:
        n_feat = len(x.columns)

    x_unstack = x.unstack(level=0)
    if fillna:
        x_unstack = x_unstack.fillna(0)

    x_3d = x_unstack.values.reshape(x_unstack.shape[0], n_feat, -1)  # inst * feat * date
    tensor = torch.from_numpy(x_3d).to(torch.float32).permute(0, 2, 1)  # Tensor with shape inst * date * feat
    return tensor


def transform_data(x_train, y_train, x_valid, y_valid, z_train=None, z_valid=None, for_cnn: bool = False,
                   for_rnn: bool = False):
    """
    将DataFrame或Series拆成list, 并根据模型类型对数据的shape进行调整
    """
    if isinstance(x_train, DataFrame) or isinstance(x_train, Series):
        if not for_rnn:
            x_train, x_valid = from_pandas_to_list(x_train, for_cnn), from_pandas_to_list(x_valid, for_cnn)
            y_train, y_valid = from_pandas_to_list(y_train), from_pandas_to_list(y_valid)
        else:
            x_train, x_valid = from_pandas_to_rnn(x_train, fillna=True), from_pandas_to_rnn(x_valid, fillna=True)
            y_train, y_valid = from_pandas_to_rnn(y_train), from_pandas_to_rnn(y_valid)
        if z_train is not None:
            z_train = from_pandas_to_list(z_train)
        if z_valid is not None:
            z_valid = from_pandas_to_list(z_valid)
    return x_train, y_train, x_valid, y_valid, z_train, z_valid


def from_series_to_edge(x: Series, threshold: float = 0.5, layout: str = "csr", shift: int = 0,
                        select_instrument: bool = True) -> list:
    """
    计算x的相关系数矩阵. 如果相关性 < threshold则两个资产没有相关性, 值为0; 若>=threshold则有相关性, 值为1

    默认返回inst_day * inst_day的矩阵, 压缩成csr格式; 如果关掉select_instrument则返回inst * inst的矩阵, inst包含所有instrument
    """
    if shift > 0:
        x = x.groupby(level=1).shift(shift).fillna(0)
    corr_matrix = x.unstack().corr().fillna(0)
    corr_matrix[abs(corr_matrix) < threshold] = 0
    corr_matrix[corr_matrix != 0] = 1
    inst = x.groupby(level=0).apply(lambda a: a.index.get_level_values(1).unique().values.tolist()).values
    mat_list = []

    for d in range(len(inst)):
        if select_instrument:
            in_col = corr_matrix.columns.isin(inst[d])
            relation_ = corr_matrix[corr_matrix.columns[in_col]]
            relation_ = relation_[relation_.index.isin(inst[d])]
        else:
            relation_ = corr_matrix
        tensor = torch.from_numpy(relation_.values).to(torch.float32)
        if layout == "csr":
            mat_list.append(tensor.to_sparse_csr())
        else:
            mat_list.append(tensor.to_sparse_coo())
    return mat_list


def calc_tensor_corr(x: Tensor, y: Tensor):
    if x.shape != y.shape:
        raise ValueError("The shapes of x and y must be the same.")
    mask = ~torch.isnan(y)
    mean_x = torch.mean(x[mask])
    mean_y = torch.mean(y[mask])
    std_x = torch.std(x[mask])
    std_y = torch.std(y[mask])
    return torch.mean((x[mask] - mean_x) * (y[mask] - mean_y)) / (std_x * std_y)


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


class style_mse(torch.nn.Module):
    def __init__(self):
        """
        当我们想控制预测值和某个变量的相关系数时

        self.loss = style_mse()
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


class robust_mse(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        """
        基于梯度设计一个新的范数, 让加上范数的mse更加鲁棒(受对抗训练的启发)

        先生成对抗样本，然后计算置信度(adv), 然后调整beta

        详情参考
        https://arxiv.org/abs/1810.09936
        和
        https://github.com/yuxiangalvin/Stock-Move-Prediction-with-Adversarial-Training-Replicate/blob/main/src/codes/pred_lstm.py
        todo: 参考以下链接实现生成样本和计算损失
        https://geek-docs.com/pytorch/pytorch-questions/169_pytorch_how_does_one_implement_adversarial_examples_in_pytorch.html
        """
        super().__init__(*args, **kwargs)


class Model(torch.nn.Module):
    def __init__(self, epochs: int = 10, loss: str = "mse_loss", lr: float = 1e-3, weight_decay: float = 5e-4,
                 dropout: float = 0.2, model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = epochs
        self.loss = loss if loss != "style_mse" else style_mse()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.model = model
        self.optimizer = None
        self.for_cnn = False
        self.for_rnn = False

    def forward(self, x, **kwargs):
        pass

    def init_model(self):
        pass

    def get_loss(self, x, y, z=None, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()
        mask = ~torch.isnan(y[:, 0])
        out = self.model(x, **kwargs)
        if isinstance(self.loss, str):
            loss = eval("f." + self.loss + "(out[mask], y[mask])")
        else:
            loss = self.loss((out[mask], y[mask], z[mask]))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def predict_(self, x, **kwargs):
        self.model.eval()
        pred = self.model(x, **kwargs)
        return pred

    @torch.no_grad()
    def test(self, x, y, z=None, **kwargs):
        mask = ~torch.isnan(y[:, 0])
        pred_ = self.predict_(x, **kwargs)
        if isinstance(self.loss, str):
            loss = eval("f." + self.loss + "(pred_[mask], y[mask])")
        else:
            loss = self.loss((pred_[mask], y[mask], z))
        return float(loss)

    def fit(self, x_train, y_train, x_valid, y_valid, z_train=None, z_valid=None, **kwargs):
        if self.model is None:
            self.init_model()

        x_train, y_train, x_valid, y_valid, z_train, z_valid = transform_data(x_train, y_train, x_valid, y_valid,
                                                                              z_train, z_valid, for_cnn=self.for_cnn,
                                                                              for_rnn=self.for_rnn)

        for epoch in range(1, self.epochs + 1):
            total_loss_train = 0
            total_loss_val = 0
            val_ic = 0
            for i in range(len(x_train)):
                loss_train = self.get_loss(x=x_train[i], y=y_train[i], z=z_train[i] if z_train is not None else None,
                                           **kwargs)
                total_loss_train += loss_train
            for i in range(len(x_valid)):
                loss_val = self.test(x=x_valid[i], y=y_valid[i], z=z_valid[i] if z_valid is not None else None,
                                     **kwargs)
                total_loss_val += loss_val
                val_ic += float(calc_tensor_corr(self.predict_(x_valid[i]), y_valid[i]))
            print("Epoch:", epoch, "loss:", total_loss_train / len(x_train), "val_loss:",
                  total_loss_val / len(x_valid), "val_ic:", val_ic / len(x_valid))

    def predict_pandas(self, x: DataFrame, **kwargs) -> Series:
        index = x.index
        x = from_pandas_to_list(x, self.for_cnn)
        result = []
        for batch in x:
            result.append(Series(self.predict_(batch, **kwargs).view(-1, )))
        series = concat(result, axis=0)
        series.index = index
        return series

    def save(self, path: str = "model.pth"):
        torch.save(self.state_dict(), path)


class MLP(Model):
    def __init__(self, input_shape: int, hidden_shape: int, output_shape: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape

        self.feature_filter = torch.nn.Linear(in_features=self.input_shape, out_features=self.hidden_shape, bias=False)
        self.hidden_layer = torch.nn.Linear(in_features=self.hidden_shape, out_features=self.hidden_shape)
        self.out_layer = torch.nn.Linear(in_features=self.hidden_shape, out_features=self.output_shape)
        self.jump_layer = torch.nn.Linear(in_features=self.input_shape, out_features=self.hidden_shape)
        self.bn = torch.nn.BatchNorm1d(self.hidden_shape)

        self.optimizer = None

    def init_model(self):
        self.model = MLP(input_shape=self.input_shape, hidden_shape=self.hidden_shape,
                         output_shape=self.output_shape, epochs=self.epochs, loss=self.loss, lr=self.lr,
                         weight_decay=self.weight_decay, dropout=self.dropout).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x, **kwargs):
        x1 = f.relu(self.jump_layer(x))
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = f.relu(self.feature_filter(x))
        x = f.dropout(x, p=self.dropout, training=self.training)

        x = f.relu(self.hidden_layer(x))

        return self.out_layer(self.bn(x + x1))


class CNN(Model):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, output_shape: int = 1, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_shape = output_shape

        self.for_cnn = True

        self.input_conv = torch.nn.Conv1d(self.input_channels, self.hidden_channels, kernel_size=1)
        self.hidden_conv = torch.nn.Conv1d(self.hidden_channels, self.output_channels, kernel_size=1)

        self.bn = torch.nn.BatchNorm1d(self.hidden_channels)
        self.bn_1 = torch.nn.BatchNorm1d(self.output_channels)
        self.flatten = torch.nn.Flatten()

        self.out_layer = torch.nn.Linear(self.output_channels, self.output_shape)

    def forward(self, x, **kwargs):
        x = f.relu(self.input_conv(x))
        x = self.bn(x)

        x = f.relu(self.hidden_conv(x))
        x = self.bn_1(x)
        x = self.flatten(x)

        x = self.out_layer(x)
        return x

    def init_model(self):
        self.model = CNN(input_channels=self.input_channels, hidden_channels=self.hidden_channels,
                         output_channels=self.output_channels, output_shape=self.output_shape, epochs=self.epochs,
                         loss=self.loss, lr=self.lr, weight_decay=self.weight_decay,
                         dropout=self.dropout).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class GRU(Model):
    def __init__(self, input_shape: int, hidden_shape: int, output_shape: int = 1, n_layers: int = 1,
                 timesteps: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        input shape: N, L, Hin, 即batch_size(=daily instrument), datetime(=timesteps), n_features
        """
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.timesteps = timesteps
        self.n_layers = n_layers

        self.gru = torch.nn.GRU(self.input_shape, self.hidden_shape, batch_first=True, bias=False,
                                num_layers=self.n_layers)
        self.bn = torch.nn.BatchNorm1d(self.hidden_shape)
        self.linear = torch.nn.Linear(self.hidden_shape, self.output_shape)

        self.for_rnn = True

    def forward(self, x, **kwargs):
        x, _ = self.gru(x)
        x = f.relu(x[:, -1, :])
        x = self.bn(x)
        x = self.linear(x)
        return x

    def init_model(self):
        self.model = GRU(input_shape=self.input_shape, hidden_shape=self.hidden_shape, output_shape=self.output_shape,
                         timesteps=self.timesteps, n_layers=self.n_layers, epochs=self.epochs, loss=self.loss,
                         lr=self.lr, weight_decay=self.weight_decay, dropout=self.dropout).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def fit(self, x_train, y_train, x_valid, y_valid, z_train=None, z_valid=None, **kwargs):
        x_train, y_train, x_valid, y_valid, z_train, z_valid = transform_data(x_train, y_train, x_valid, y_valid,
                                                                              z_train, z_valid, for_cnn=self.for_cnn,
                                                                              for_rnn=self.for_rnn)
        if self.model is None:
            self.init_model()
        for epoch in range(1, self.epochs + 1):
            total_loss_train = 0
            total_loss_val = 0
            val_ic = 0
            for i in range(0, x_train.shape[1] - self.timesteps):
                loss_train = self.get_loss(x_train[:, i:i + self.timesteps, :], y_train[:, self.timesteps + i, :],
                                           z_train[self.timesteps + i] if z_train is not None else None)
                total_loss_train += loss_train
            for i in range(0, x_valid.shape[1] - self.timesteps):
                loss_val = self.test(x_valid[:, i:i + self.timesteps, :], y_valid[:, self.timesteps + i, :],
                                     z_valid[self.timesteps + i] if z_valid is not None else None)
                total_loss_val += loss_val
                val_ic += float(calc_tensor_corr(self.predict_(x_valid[:, i:i + self.timesteps, :]),
                                                 y_valid[:, self.timesteps + i, :]))
            print("Epoch:", epoch, "loss:", total_loss_train / len(x_train), "val_loss:",
                  total_loss_val / len(x_valid), "val_ic:", val_ic / len(x_valid))

    def predict_pandas(self, x: DataFrame, **kwargs) -> Series:
        x_tensor = from_pandas_to_rnn(x, fillna=True)
        result = []
        for i in range(x_tensor.shape[1] - self.timesteps):
            result.append(Series(self.predict_(x_tensor[:, i:i + self.timesteps, :], **kwargs).view(-1, )))

        days = x.index.get_level_values(0).unique()[-len(result):]
        instrument = x.index.get_level_values(1).unique().to_list()
        name_0, name_1 = x.index.names[0], x.index.names[1]
        predict = []
        for i in range(len(result)):
            df = DataFrame(result[i])
            df[name_0] = days[i]
            df[name_1] = instrument
            predict.append(df.set_index([name_0, name_1]).iloc[:, 0])
        predict = concat(predict, axis=0)
        return predict[predict.index.isin(x.index)]


class GNN(Model):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, output_shape: int = 1,
                 aggr="mean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_shape = output_shape
        self.aggr = aggr

        self.bn_1 = torch.nn.BatchNorm1d(self.output_channels)
        self.bn_2 = torch.nn.BatchNorm1d(self.hidden_channels)
        self.bn_3 = torch.nn.BatchNorm1d(self.output_channels)

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = torch.from_numpy(np.zeros(shape=(x.shape[0], x.shape[0]))).to(torch.float32).to_sparse_coo()

        x1 = f.relu(self.input_conv_1(x, edge_index))  # in_channels, out_channels
        x1 = self.bn_1(x1)
        # x1 = f.dropout(x1, p=self.dropout, training=self.training)

        x2 = f.relu(self.input_conv_2(x, edge_index))  # in_channels, hidden_channels
        x2 = self.bn_2(x2)

        x3 = f.relu(self.linear_1(x))  # in_channels, hidden_channels
        x3 = f.dropout(x2, p=self.dropout, training=self.training)

        x3 = f.relu(self.linear_2(x3 + x2))  # hidden_channels, out_channels
        x3 = self.bn_3(x3)

        x = self.output_layer(x3 + x1)
        return x

    def init_model(self):
        pass

    def fit(self, x_train, y_train, x_valid, y_valid, z_train=None, z_valid=None, edge_train: list = None,
            edge_valid: list = None):
        if self.model is None:
            self.init_model()

        x_train, y_train, x_valid, y_valid, z_train, z_valid = transform_data(x_train, y_train, x_valid, y_valid,
                                                                              z_train, z_valid, for_cnn=self.for_cnn,
                                                                              for_rnn=self.for_rnn)
        for epoch in range(1, self.epochs + 1):
            total_loss_train = 0
            total_loss_val = 0
            val_ic = 0
            for i in range(len(x_train)):
                loss_train = self.get_loss(x=x_train[i], y=y_train[i], z=z_train[i] if z_train is not None else None,
                                           edge_index=edge_train[i] if edge_train is not None else None)
                total_loss_train += loss_train
            for i in range(len(x_valid)):
                loss_val = self.test(x=x_valid[i], y=y_valid[i], z=z_valid[i] if z_valid is not None else None,
                                     edge_index=edge_valid[i] if edge_valid is not None else None)
                total_loss_val += loss_val
                val_ic += float(calc_tensor_corr(self.predict_(x_valid[i]), y_valid[i]))
            print("Epoch:", epoch, "loss:", total_loss_train / len(x_train), "val_loss:",
                  total_loss_val / len(x_valid), "val_ic:", val_ic / len(x_valid))

    def predict_pandas(self, x: DataFrame, edge_index=None) -> Series:
        index = x.index
        x = from_pandas_to_list(x, self.for_cnn)
        result = []
        for i in range(len(x)):
            result.append(
                Series(self.predict_(x[i], edge_index=edge_index[i] if edge_index is not None else None).view(-1, )))
        series = concat(result, axis=0)
        series.index = index
        return series


class GraphSage(GNN):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, *args, **kwargs):
        """
        a simple example:

        edges = models.from_series_to_edge(y_train)  # 主要多了这步, 需要自己构建edge_index
        model = models.GraphSage(input_channels=50, hidden_channels=20, output_channels=10)
        model.fit(x_train, y_train, x_valid, y_valid, edge_index=edges)
        pred = model.predict_pandas(x_test, edge_index=edges)
        """
        super().__init__(input_channels, hidden_channels, output_channels, *args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.input_conv_1 = SAGEConv(in_channels=self.input_channels, out_channels=self.output_channels, aggr=self.aggr)
        self.input_conv_2 = SAGEConv(in_channels=self.input_channels, out_channels=self.hidden_channels, aggr=self.aggr)
        self.linear_1 = torch.nn.Linear(self.input_channels, self.hidden_channels)
        self.linear_2 = torch.nn.Linear(self.hidden_channels, self.output_channels)
        self.output_layer = torch.nn.Linear(self.output_channels, self.output_shape)

    def init_model(self):
        self.model = GraphSage(input_channels=self.input_channels, hidden_channels=self.hidden_channels,
                               output_channels=self.output_channels, output_shape=self.output_shape, aggr=self.aggr,
                               epochs=self.epochs, loss=self.loss, lr=self.lr, weight_decay=self.weight_decay,
                               dropout=self.dropout).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class GCN(GNN):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, *args, **kwargs):
        """
        注意GCN的edge必须非负, 因为D^0.5的存在. 详见
        https://github.com/pyg-team/pytorch_geometric/issues/61
        """
        super().__init__(input_channels, hidden_channels, output_channels, *args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.input_conv_1 = GCNConv(in_channels=self.input_channels, out_channels=self.output_channels, aggr=self.aggr)
        self.input_conv_2 = GCNConv(in_channels=self.input_channels, out_channels=self.hidden_channels, aggr=self.aggr)
        self.linear_1 = torch.nn.Linear(self.input_channels, self.hidden_channels)
        self.linear_2 = torch.nn.Linear(self.hidden_channels, self.output_channels)
        self.output_layer = torch.nn.Linear(self.output_channels, self.output_shape)

    def init_model(self):
        self.model = GCN(input_channels=self.input_channels, hidden_channels=self.hidden_channels,
                         output_channels=self.output_channels, output_shape=self.output_shape, aggr=self.aggr,
                         epochs=self.epochs, loss=self.loss, lr=self.lr, weight_decay=self.weight_decay,
                         dropout=self.dropout).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class GAT(GNN):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, *args, **kwargs):
        super().__init__(input_channels, hidden_channels, output_channels, *args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.input_conv_1 = GATConv(in_channels=self.input_channels, out_channels=self.output_channels, aggr=self.aggr)
        self.input_conv_2 = GATConv(in_channels=self.input_channels, out_channels=self.hidden_channels, aggr=self.aggr)
        self.linear_1 = torch.nn.Linear(self.input_channels, self.hidden_channels)
        self.linear_2 = torch.nn.Linear(self.hidden_channels, self.output_channels)
        self.output_layer = torch.nn.Linear(self.output_channels, self.output_shape)

    def init_model(self):
        self.model = GAT(input_channels=self.input_channels, hidden_channels=self.hidden_channels,
                         output_channels=self.output_channels, output_shape=self.output_shape, aggr=self.aggr,
                         epochs=self.epochs, loss=self.loss, lr=self.lr, weight_decay=self.weight_decay,
                         dropout=self.dropout).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class Hybrid(GNN):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, timestep: int = 1,
                 n_gru: int = 2, *args, **kwargs):
        super().__init__(input_channels, hidden_channels, output_channels, *args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.timestep = timestep
        self.n_gru = n_gru

        self.gru_1 = torch.nn.GRU(self.input_channels, self.input_channels, batch_first=True,
                                    num_layers=self.n_gru)
        self.gru_2 = torch.nn.GRU(self.input_channels, self.output_channels, batch_first=True,
                                  num_layers=self.n_gru)
        self.bn_1 = torch.nn.BatchNorm1d(self.input_channels)

        self.gnn_1 = SAGEConv(self.input_channels, self.hidden_channels, aggr=self.aggr)
        self.gnn_2 = SAGEConv(self.input_channels, self.output_channels, aggr=self.aggr)
        self.bn_2 = torch.nn.BatchNorm1d(self.output_channels)
        self.bn_3 = torch.nn.BatchNorm1d(self.output_channels)
        self.bn_4 = torch.nn.BatchNorm1d(self.output_channels)
        self.linear = torch.nn.Linear(self.hidden_channels, self.output_channels)
        self.output_layer = torch.nn.Linear(self.output_channels, self.output_shape)

    def forward(self, x, edge_index=None):
        # x: inst * feat
        x_for_gru = x.reshape(x.shape[0], x.shape[1], 1).permute(0, 2, 1)
        if edge_index is None:
            edge_index = torch.from_numpy(np.zeros(shape=(x.shape[0], x.shape[0]))).to(torch.float32).to_sparse_coo()

        x1, _ = self.gru_1(x_for_gru)
        x1 = f.relu(x1[:, -1, :])
        x1 = self.bn_1(x1)

        x2 = self.gnn_1(x + x1, edge_index)  # gnn with gru's output as input
        x2 = f.relu(x2)
        x2 = self.linear(x2)
        x2 = f.relu(x2)
        x2 = self.bn_2(x2)

        x3 = self.gnn_2(x, edge_index)  # pure gnn
        x3 = f.relu(x3)
        x3 = self.bn_3(x3)

        x4, _ = self.gru_2(x_for_gru)
        x4 = f.relu(x4[:, -1, :])
        x4 = self.bn_4(x4)
        return self.output_layer(x2 + x3 + x4)

    def init_model(self):
        self.model = Hybrid(input_channels=self.input_channels, hidden_channels=self.hidden_channels,
                            output_channels=self.output_channels, output_shape=self.output_shape, aggr=self.aggr,
                            epochs=self.epochs, loss=self.loss, lr=self.lr, weight_decay=self.weight_decay,
                            dropout=self.dropout, timestep=self.timestep, n_gru=self.n_gru).to(torch.float32)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
