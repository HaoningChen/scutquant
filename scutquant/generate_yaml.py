import yaml

data_kwargs = {
    "format": "csv",
    "index_col": ["datetime", "ts_code"],
    "index_names": None,
    "process_nan": True,
    "label": {
        "by": "pct_chg",
        "divide": 100
    }
}
factor_kwargs = {
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "vol",
    "amount": "amount",
    "windows": [5, 10, 20, 30, 60],
    "fillna": False
}
process_kwargs = {
    "groupby": "code",
    "split": {
        "test_start_date": "2019-01-01",
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
    "volume": "vol"
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
