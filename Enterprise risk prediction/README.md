# [CCF-企业经营退出风险预测](http://www.datafountain.cn/#/competitions/271/activity) 初赛方案
## 赛题介绍
参赛者需要利用训练数据集中企业信息数据，构建算法模型，并利用该算法模型对验证数据集中企业，给出预测结果以及风险概率值

[竞赛概述](http://www.datafountain.cn/?u=7586043&&#/competitions/271/intro)
## 数据集
数据集介绍：[企业经营退出风险预测数据集](http://www.datafountain.cn/?u=7586043&&#/competitions/271/data-intro)
## 数据清洗以及特征构造
根据给的几个不同维度的数据，首先对数据进行了一些探索分析，然后分别从不同的维度构造了一些特征
- [Exploratory analysis & Preprocessing.ipynb](https://github.com/TFeiMa/Data-competition/blob/master/Enterprise%20risk%20prediction/Exploratory%20analysis%20%26%20Preprocessing.ipynb)
- [data_precessing.ipynb](https://github.com/TFeiMa/Data-competition/blob/master/Enterprise%20risk%20prediction/data_precessing.ipynb) 
## 模型尝试以及模型调参
- [model_test.ipynb](https://github.com/TFeiMa/Data-competition/blob/master/Enterprise%20risk%20prediction/Model_test.ipynb),使用网格搜索法调整模型参数，并尝试了XGBoost、gbdt、RandomFrest等模型
- [model_stacking.py](https://github.com/TFeiMa/Data-competition/blob/master/Enterprise%20risk%20prediction/model_stacking.py)一个stacking尝试
