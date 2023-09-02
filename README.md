# scutquant
[华南理工大学量化投资协会](https://github.com/ksDreamer/gzic.online/blob/main/organizations.md#%E9%87%8F%E5%8C%96%E6%8A%95%E8%B5%84%E5%8D%8F%E4%BC%9A)（华工量协）欢迎您

代码和介绍见[scutquant](https://github.com/HaoningChen/ScutQuant/tree/main/scutquant)

纯代码版本(方便下载)请转到[scutquant_pure](https://github.com/chn489/scutquant_pure)


如果有任何意见和建议，欢迎在issue区留言或者直接提pr! 如果觉得本仓库对您有用, 请点一个免费的star!  

## 导航  
[scutquant](https://github.com/HaoningChen/ScutQuant/tree/main/scutquant):  
量化工具包, 拥有一个离线量化投资平台所需的所有代码  

[实践案例(内含demo)](https://github.com/HaoningChen/ScutQuant/tree/main/实践案例): 实践出真知, 提供难度适中的任务给大家练手

[教程](https://github.com/HaoningChen/ScutQuant/blob/main/%E5%AE%9E%E8%B7%B5%E6%A1%88%E4%BE%8B/tutorial.ipynb): 演示如何使用scutquant完成从数据分析到回测的全流程  

[文件](https://github.com/HaoningChen/ScutQuant/tree/main/文件): 存放技术沙龙文件和其它资料  

联系方式(目前): harley_chen@foxmail.com

## Quant With YAML:  
当你已经对量化投资有所了解, 又不愿意花费时间学习新的框架时, 可以在配置好[all_kwargs.yaml](https://github.com/HaoningChen/scutquant/blob/main/%E5%AE%9E%E8%B7%B5%E6%A1%88%E4%BE%8B/all_kwargs.yaml)和data文件后, 使用[pipeline](https://github.com/HaoningChen/scutquant/blob/main/%E5%AE%9E%E8%B7%B5%E6%A1%88%E4%BE%8B/quant_with_yaml.ipynb)轻松进行量化研究. 源代码请参考[scutquant.Pipeline](https://github.com/HaoningChen/scutquant/blob/main/scutquant/Pipeline.py)

另外, 如果仅需研究因子IC, 可以在配置好[factors_ana.yaml](https://github.com/HaoningChen/scutquant/blob/main/%E6%8A%80%E6%9C%AF%E5%9B%A0%E5%AD%90%E6%B5%8B%E8%AF%95/factors_ana.yaml)和data文件后, 使用[all_factors_ana](https://github.com/HaoningChen/scutquant/blob/main/%E6%8A%80%E6%9C%AF%E5%9B%A0%E5%AD%90%E6%B5%8B%E8%AF%95/factors_ana.ipynb)获取所有因子的IC, 以及查看在测试集上的所有因子IC加权均值和分层效应. 

可以根据给出的yaml示例, 修改参数以适应不同的数据集和任务需求

## 环境要求  
(最好有) Anaconda3   
python3.8 及以上（如果不使用data模块，那么3.7及以上即可）    
Windows10 64位; Ubuntu 20.04.3(20及以上应该都可以)  
[requirements.txt](https://github.com/HaoningChen/ScutQuant/blob/main/scutquant/requirements.txt)

## 如何安装  
由于暂时还没发布到PyPI(还没注册账号), 所以目前只能从github下载压缩包并解压到site_packages文件夹. 推荐从[scutquant_pure](https://github.com/chn489/scutquant_pure)下载(解压后需要将文件夹名字改成scutquant)
