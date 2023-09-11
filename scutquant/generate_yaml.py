"""
以下代码旨在提供一个yaml的示例

一个完整的量化研究流程分为
(1) 读取数据集, 设置索引, 设置目标值
(2) 生成因子
(3) 处理feature和label, 包括数据清洗, 标准化等, 并拆分数据集为训练集, 验证集合测试集三部分
(4) 模型(用于信号合成)
(5) 处理用于回测的数据集
(6) 设置回测参数, 包括策略, 费用等等, 参考Pipeline.backtest函数
(7) 报告回测结果

前5步分别对应下面的 data_kwargs, factor_kwargs, process_kwargs, fit_kwargs, prepare, 最后将这些kwargs合并成一个大的字典
all_kwargs, 并保存到指定的文件夹下, 最后调用Pipeline.pipeline读取all_kwargs.yaml和data, 并根据kwargs的内容自动执行量化研究

更多细节请参考Pipeline
"""

import yaml

data_kwargs = {
    "format": "csv",
    "index_col": ["datetime", "instrument"],
    "index_names": None,
    "process_nan": True,
    "label": {
        "by": "pct_chg",
        "divide": 100
    }
}
factor_kwargs = {
    "normalize": False,
    "fill": False,
    "windows": [5, 10, 20, 30, 60]
}
process_kwargs = {
    "groupby": "instrument",
    "split": {
        "test_start_date": "2017-01-01",
        "split_method": "split",
        "split_kwargs": {
            "train": 0.7,
            "valid": 0.3
        }
    }
}

fit_kwargs = {
    "model": "Ensemble",
    "params": {
        "epochs": 15,
    },
    "save": True
}

prepare = {
    "deal_price": "close",
    "volume": "volume"
}

all_kwargs = {
    "data": data_kwargs,
    "factor": factor_kwargs,
    "process": process_kwargs,
    "fit": fit_kwargs,
    "prepare": prepare,
    "backtest": None,
    "report": None
}

with open('D:/Desktop/workflow/all_kwargs.yaml', 'w') as f:
    yaml.dump(all_kwargs, f)
f.close()
