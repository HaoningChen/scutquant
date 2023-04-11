from keras import layers, regularizers, models
from tensorflow import keras

"""
    example:

    dnn = DNN(epochs=1)
    dnn.fit(x_train, y_train, x_valid, y_valid)  # 训练模型
    pred = dnn.predict(x_test)  # 获取预测值
    model = dnn.model  # 获取模型(用于保存和部署)

    其它模型同理
    
    目前实现的模型包括: DNN, CNN, LSTM, Bi-LSTM和Attention (scutquant模块还有OLS, ridge, lasso, XGBoost, hybrid和lightGBM)
    计划实现的模型包括: RNN, GRU
"""


class DNN:
    def __init__(self, n_layers=2, activation="swish", optimizer="adam", loss="mse", metrics=None, l1=1e-5, l2=1e-5,
                 epochs=10, batch_size=256, model=None):
        if metrics is None:
            metrics = ["mae", "mape"]
        self.layers = n_layers
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model

    def create_model(self, x_input):
        shape = x_input.shape[1]
        inputs = keras.Input(shape=(shape,))
        x = layers.Dense(shape, activation=self.activation)(inputs)
        for i in range(self.layers - 1):
            x = layers.Dense(1024, activation=self.activation,
                             kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2))(x)
        x = layers.Dense(256, activation=self.activation,
                         kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2))(x)
        output = layers.Dense(1, activation=self.activation,
                              kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2))(x)

        model = keras.Model(inputs, output)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.model = DNN.create_model(self, x_train)
        self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=self.epochs,
                       batch_size=self.batch_size)

    def predict(self, x_test):
        predict = self.model.predict(x_test)
        return predict


class LSTM:
    def __init__(self, n_layers=2, activation="swish", optimizer="adam", loss="mse", metrics=None, l1=1e-5, l2=1e-5,
                 epochs=10, batch_size=256, model=None):
        if metrics is None:
            metrics = ["mae", "mape"]
        self.layers = n_layers
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model

    def create_model(self, x_input):
        shape = x_input.shape[1]
        inputs = keras.Input(shape=(shape,))

        x = layers.Reshape((-1, shape,))(inputs)
        if self.layers > 1:
            for i in range(self.layers - 1):
                x = layers.LSTM(shape, kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
                                return_sequences=True)(x)
        x = layers.LSTM(shape, kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
                        return_sequences=False)(x)
        output = layers.Dense(1, activation=self.activation,
                              kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2))(x)
        model = keras.Model(inputs, output)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.model = LSTM.create_model(self, x_train)
        self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=self.epochs,
                       batch_size=self.batch_size)

    def predict(self, x_test):
        predict = self.model.predict(x_test)
        return predict


class Bi_LSTM:
    def __init__(self, n_layers=2, activation="swish", optimizer="adam", loss="mse", metrics=None, l1=1e-5, l2=1e-5,
                 epochs=10, batch_size=256, model=None):
        if metrics is None:
            metrics = ["mae", "mape"]
        self.layers = n_layers
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model

    def create_model(self, x_input):
        shape = x_input.shape[1]
        inputs = keras.Input(shape=(shape,))

        x = layers.Reshape((-1, shape,))(inputs)
        if self.layers > 1:
            for i in range(self.layers - 1):
                x = layers.Bidirectional(
                    layers.LSTM(shape, kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
                                return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(shape, kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
                                             return_sequences=False))(x)
        output = layers.Dense(1, activation=self.activation,
                              kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2))(x)
        model = keras.Model(inputs, output)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.model = Bi_LSTM.create_model(self, x_train)
        self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=self.epochs,
                       batch_size=self.batch_size)

    def predict(self, x_test):
        predict = self.model.predict(x_test)
        return predict


class Attention:
    """
    复现了Attention Is All You Need 中的部分结构(指encoder, decoder目前还没实现)
    """

    def __init__(self, n_attentions=8, n_encoders=1, activation="swish", optimizer="adam", loss="mse", metrics=None,
                 l1=1e-5, l2=1e-5, epochs=10, batch_size=256, model=None):
        if metrics is None:
            metrics = ["mae", "mape"]
        self.att = n_attentions
        self.encoders = n_encoders
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model

    def encoder(self, query, value):
        # 选择用多个attention取简单平均, 而不是Multi-Head Attention(以后或许会修改)
        att = []
        for i in range(0, self.att):
            att.append(layers.Attention()([query, value]))
        weight_average_attention = sum(att) / self.att
        ln_1 = layers.LayerNormalization()(weight_average_attention + query + value)  # 残差连接
        # 下面省略了全连接部分，因为发现加上全连接层效果不好
        return ln_1

    def create_model(self, x_input):
        shape = x_input.shape[1]
        inputs = keras.Input(shape=(shape,))

        q = layers.Dense(1024, activation=self.activation,
                         kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2))(inputs)
        v = layers.Dense(1024, activation=self.activation,
                         kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2))(inputs)

        encoders = []
        for i in range(self.encoders):
            encoders.append(Attention.encoder(self, q, v))

        flatten = layers.Flatten()(sum(encoders))
        output = layers.Dense(1, kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2))(flatten)
        model = keras.Model(inputs=inputs, outputs=output)

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.model = Attention.create_model(self, x_train)
        self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=self.epochs,
                       batch_size=self.batch_size)

    def predict(self, x_test):
        predict = self.model.predict(x_test)
        return predict


class CNN:
    def __init__(self, n_layers=2, filters=32, kernel_size=9, strides=3, activation="swish", optimizer="adam",
                 loss="mse", metrics=None, l1=1e-5, l2=1e-5, epochs=10, batch_size=256, model=None):
        if metrics is None:
            metrics = ["mae", "mape"]
        self.layers = n_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model

    def create_model(self):
        model = keras.Sequential()
        n_filters = self.filters
        kernel_size = self.kernel_size
        strides = self.strides

        if self.layers > 1:
            for i in range(self.layers - 1):
                model.add(
                    layers.Conv1D(n_filters, kernel_size=kernel_size, strides=strides, activation=self.activation))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling1D(pool_size=2, strides=1))
                n_filters *= 2
                kernel_size = kernel_size - 2 if kernel_size > 2 else 1

        model.add(
            layers.Conv1D(n_filters, kernel_size=kernel_size, strides=strides, activation=self.activation))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2, strides=1))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, kernel_regularizer=regularizers.l1_l2(self.l1, self.l2)))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def fit(self, x_train, y_train, x_valid, y_valid):
        x_train, x_valid = x_train.values.reshape(-1, x_train.shape[1], 1), \
                           x_valid.values.reshape(-1, x_valid.shape[1], 1)
        self.model = CNN.create_model(self)
        self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=self.epochs,
                       batch_size=self.batch_size)

    def predict(self, x_test):
        x_test = x_test.values.reshape(-1, x_test.shape[1], 1)
        predict = self.model.predict(x_test)
        return predict


class Ensemble:
    # https://www.kaggle.com/competitions/ubiquant-market-prediction结果表明, 多个神经网络的ensemble要比单个的效果更好
    def __init__(self, model=None, weight=None):
        """
        :param model: None 或者已训练好的模型列表. 如果为None则在fit()函数中使用默认参数训练一个CNN模型和一个Bi-LSTM模型
        :param weight: None 或者各模型的权重
        """
        self.models = model
        self.weight = weight

    def create_model(self, x_input):
        if self.models is None:
            self.models = []
            self.models.append(CNN())
            self.models.append((Bi_LSTM()))
            self.models[0].create_model()
            self.models[1].create_model(x_input)
            w = 1 / len(self.models)
            self.weight = [w for _ in range(len(self.models))] if self.weight is None else self.weight

    def fit(self, X_train, y_train, X_valid, y_valid):
        Ensemble.create_model(self, X_train)
        for i in range(len(self.models)):
            self.models[i].fit(X_train, y_train, X_valid, y_valid)

    def predict(self, X_test):
        predict = {}
        for i in range(len(self.models)):
            pred = self.models[i].predict(X_test)
            predict[str(i)] = pred * self.weight[i]
        return sum(predict[k] for k in predict.keys())

    def save(self, target_dir=''):
        for i in range(len(self.models)):
            self.models[i].model.save_model(target_dir + "model" + str(i))

    def load(self, n_models, target_dir=''):
        for i in range(n_models):
            model = models.load_model(target_dir + "model" + str(i))
            self.models.append(model)
