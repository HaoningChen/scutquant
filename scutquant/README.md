## 快速上手  
请查看[tutorial.ipynb](https://github.com/HaoningChen/ScutQuant/blob/main/%E5%AE%9E%E8%B7%B5%E6%A1%88%E4%BE%8B/tutorial.ipynb)以快速上手  

## 各模块介绍(按照量化研究的流程介绍)  

### [data](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/data.py):   
获取数据模块, 基于akshare实现. 目前功能只有获取指数成分股数据  
### [alpha](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/alpha.py): 
因子模块, 实现了两个因子库, 一个由alpha360()函数实现, 另一个由make_factors实现.  
前者是基础价量数据的滞后项, 作除单位处理; 后者是技术因子, 目前包含158个较为有效的因子
### [scutquant](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/scutquant.py):  
包含了scutquant包最核心的数据分析和数据处理功能, 还有一些快速建模函数(支持一键训练线性回归模型(基于sklearn), hybrid模型和lightGBM)   
### [models](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/models.py):  
使用tensorflow和keras写的模型, 可以方便地一键训练自定义的神经网络. 目前已实现的网络有DNN, LSTM, Bi-LSTM和Attention  
### [account](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/account.py):  
账户, 回测时使用虚拟账户进行仿真交易  
### [strategy](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/strategy.py):  
策略, 可以继承BaseStrategy自己设计策略  
### [signal_generator](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/signal_generator.py):  
根据模型预测值和策略, 在每一个时间t, 生成下单指令order和交易价格current_price
### [executor](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/executor.py):  
在account, strategy和signal_generator的基础上进行封装而成的执行器, 负责策略的执行和更新account  
### [report](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/report.py):  
报告回测结果, 也可用于测试因子和模型预测值的单调性(见group_return_ana()函数)  

各部分基本上都能单独拿出来使用, 耦合度非常低(在底层代码基础上封装的executor和需要实时计算因子值的signal_generator模块除外)

## Workflow  
![1](https://user-images.githubusercontent.com/101194077/210078909-6f1f7381-68cf-4765-8c8c-5feaf68f39c0.png)


## 数据处理流程（到执行回测前）  
![2](https://user-images.githubusercontent.com/101194077/209441805-ecee94f8-794a-4431-819f-73f66d182aef.png)
