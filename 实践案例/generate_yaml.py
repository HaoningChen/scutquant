import yaml

data_kwargs = {
    "format": "csv",
    "index_col": ["datetime", "ts_code"],
    "index_names": ["datetime", "instrument"],
    "process_nan": True,
    "label": {
        "by": "pct_chg",
        # "shift1":,
        # "shift2":,
        "divide": 100
    }
}

factor_kwargs = None

process_kwargs = {
    "label": "label",
    "groupby": "stock_code",
    "split": {
        "test_start_date": "2019-01-01",
        "split_method": "split",
        "split_kwargs": {
            "train": 0.7,
            "valid": 0.3
        }
    },
    "normalization": "z",
    "clip_data": 3,
    "label_norm": True,
    "select_features": False,
    "orth_data": False,
}

fit_kwargs = {
    "model": "Ensemble",
    "params": {
        "epochs": 15
    },
    "save": True
}

prepare_kwargs = {
    "deal_price": "close",
    "volume": "vol"
}

backtest_kwargs = None

report_kwargs = None

all_kwargs = {
    "data": data_kwargs,
    "factor": factor_kwargs,
    "process": process_kwargs,
    "fit": fit_kwargs,
    "prepare": prepare_kwargs,
    "backtest": backtest_kwargs,
    "report": report_kwargs
}

with open('D:/Desktop/workflow/all_kwargs.yaml', 'w') as f:
    yaml.dump(all_kwargs, f)
f.close()
