该文件夹用于存放库和案例  


目前的库主要有scutquant，包括数据清洗（计算缺失值百分比、处理缺失值、计算0或某个值的占比、降采样和升采样（基于bootstrap））、特征工程（三种标准化方式、计算多重共线性、PCA和对称正交、计算feature和label的Mutual Information、画出数据分布）、拆分数据集（按顺序拆分、随机拆分（train_test_split二次封装）和GroupKFold（尚未实现））等功能，还有一个集成了上述功能的AutoProcessor，新手可以调用它一键处理数据；以及一个自动建模函数（目前只有线性回归模型），具体使用案例见 scutquant_demo.ipynb


目前还计划加入时间序列分析模块，并完善自动建模功能
