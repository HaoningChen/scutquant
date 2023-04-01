## 关于测试  
股票池: 沪深300  
测试时间: 2006-01-01~2022-11-30(剔除缺失值)  

## 结论  
记录了截止当前版本的技术因子表现(源代码见[scutquant.alpha.make_factors()](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/alpha.py))  
因子构造有一部分参考了qlib的alpha158(复现了其中79个), 并把alpha158的表达式转化成显性的python代码(虽然qlib的算子确实方便了构建因子, 但是耦合度太高了)  
在因子构造时加入截面的信息(例如IDX系列)可能会提高因子质量
