import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import metrics
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
# from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from time import time
#-----------------------------------------------训练---------------------------------------------------------#
# 开始时间
t0 = time()


# 读取数据集
data_train = pd.read_csv('data_analysis/data_train.csv',encoding='gb2312')

targets = data_train['TARGET']
train_data = data_train.drop(labels=['EID','TARGET'],axis=1)

#  划分样本集
train_x,test_x,train_y,test_y = train_test_split(train_data,targets,test_size=0.5,random_state=66)

# 设置参数
xgb = XGBClassifier(n_estimators=300,max_depth=5,nthread=20,scale_pos_weight=4,learning_rate=0.07)
# 特征选择
rfecv = RFECV(estimator=xgb, step=10, cv=StratifiedKFold(3),n_jobs =20,
              scoring='roc_auc')
rfecv.fit(train_x, train_y)

pre_y = rfecv.predict_proba(test_x)[:,1]
pre_y_categ = rfecv.predict(test_x)
# 计算auc
fpr, tpr, thresholds = metrics.roc_curve(test_y, pre_y)
auc=metrics.auc(fpr, tpr)
f1 = metrics.f1_score(test_y,pre_y_categ)
print("AUC得分为：")
print(auc)
print('f1-score为：')
print(f1)
print("Optimal number of features :" )
print(rfecv.ranking_ )
print('n_features_')
print(rfecv.n_features_)
print('support_')
print(rfecv.support_)
total_time = time() - t0
print("运行时间为：%f"%total_time)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
