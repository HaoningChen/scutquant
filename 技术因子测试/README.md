## 关于测试  
股票池: 沪深300  
测试时间: 2006-01-01~2023-06-20(剔除缺失值)  

## 结论  
记录了截止当前版本的技术因子表现(源代码见[scutquant.alpha.qlib158()](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/alpha.py))  
Qlib的alpha158的性能主要来自除单位的操作(对于price类因子除以收盘价, 对于volume类因子除以交易量), 换言之, 因子表达式本身不见得有多高明
