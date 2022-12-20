**任务：**  
1、读取任意金融数据集，设置索引，处理缺失值  
2、自己构建特征(即因子)，拆分训练集和测试集，完成特征工程(至少应包括标准化数据集)和数据清洗  
3、搭建至少5种模型，并评价拟合优度

**教程：**  
[tutorial](https://github.com/HaoningChen/ScutQuant/blob/main/%E5%AE%9E%E8%B7%B5%E6%A1%88%E4%BE%8B/tutorial.ipynb)是一个使用沪深300数据集(日频数据, 时间从2014-01-01到20021-06-11, 剔除ST股)构建因子, 并使用线性回归模型(OLS)预测收益率的例子，要复现它请使用最新版本的scutquant.  
[教程数据](https://www.kaggle.com/datasets/harleychan/csi300)

**PS:模型的分类(从形式的角度)**  
**[线性模型](https://scikit-learn.org/stable/modules/linear_model.html):** 线性回归模型, 包括OLS、ridge回归和lasso回归，弹性网络模型等；分类模型有逻辑回归等;  
**树模型:** [xgboost](https://www.kaggle.com/code/alexisbcook/xgboost), [catboost](https://catboost.ai/), [lightgbm](https://lightgbm.readthedocs.io/en/v3.3.2/)...  
**神经网络模型:** DNN, [CNN](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf), [TCN](https://arxiv.org/abs/1803.01271), RNN, [LSTM](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext), [Bi-LSTM](https://www.researchgate.net/publication/306093736_Attention-Based_Bidirectional_Long_Short-Term_Memory_Networks_for_Relation_Classification), [Transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)..., 神经网络原理请参考[链接](https://github.com/microsoft/ai-edu/tree/master/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86)  
组合模型, 可以由上述三大类模型之间的任两类(或者三类同时用上)组合而成  
其它机器学习模型
