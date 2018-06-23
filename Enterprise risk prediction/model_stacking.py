import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn import preprocessing
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from heamy.estimator import Regressor, Classifier
from heamy.pipeline import ModelsPipeline
from heamy.dataset import Dataset
from sklearn.svm import SVC

#---------------------------------------------------------读取数据集-------------------------------#
data_train = pd.read_csv('data_analysis/data_train.csv',encoding='gb2312')
targets = data_train['TARGET']
train_data = data_train.drop(labels=['TARGET'],axis=1)

data_test = pd.read_csv('data_analysis/data_test.csv',encoding='gb2312')

test_data = data_test.drop(labels=['FORTARGET','PROB'],axis=1)
# ------------------------------------------------------- 划分样本集-----------------------------------#
# train_x,test_x,train_y,test_y = train_test_split(train_data,targets,test_size=0.5,random_state=66)
# create dataset
# dataset = Dataset(train_data,targets,test_data)
dataset = Dataset(train_data,targets,test_data)
#xgb = XGBClassifier(n_estimators = 1350,scale_pos_weight=4,nthread=-1,seed=6,max_depth=3,min_child_weight=6,learning_rate=0.05,
#							gamma=0,subsample=0.9,colsample_bytree=0.9,reg_alpha=8)
#--------------------------------------------------------stacking model----------------------#
model_rf1 = Classifier(dataset=dataset, estimator=RandomForestClassifier, parameters={'n_estimators': 1000,'max_depth':19,
							'criterion':'entropy','min_samples_split':15,'n_jobs':-1},name='rf1')
model_rf2 = Classifier(dataset=dataset, estimator=RandomForestClassifier, parameters={'n_estimators': 1000,'max_depth':19,
							'criterion':'gini','min_samples_split':15,'n_jobs':-1},name='rf2')
							
model_gdbt1 = Classifier(dataset=dataset, estimator=GradientBoostingClassifier, parameters={'n_estimators':600,'loss' : 'exponential',
							'max_depth':4,'min_samples_split':10,'min_weight_fraction_leaf':0.01,'learning_rate':0.06,'random_state':1},
							name='gdbt1')
model_gdbt2 = Classifier(dataset=dataset, estimator=GradientBoostingClassifier, parameters={'n_estimators':600,'loss' : 'exponential',
							'max_depth':4,'min_samples_split':10,'min_weight_fraction_leaf':0.01,'learning_rate':0.07,'random_state':2},
							name='gdbt2')
model_gdbt3 = Classifier(dataset=dataset, estimator=GradientBoostingClassifier, parameters={'n_estimators':600,'loss' : 'deviance',
							'max_depth':4,'min_samples_split':10,'min_weight_fraction_leaf':0.01,'learning_rate':0.07,'random_state':3},
							name='gdbt3')							
model_xgbt = Classifier(dataset=dataset, estimator=GradientBoostingClassifier, parameters={'n_estimators' :1350,
							'nthread':-1,'max_depth':3,'min_child_weight':6,'learning_rate':0.05,
							'gamma':0,'subsample':0.9,'colsample_bytree':0.9,'reg_alpha':8,},name='xgbt')
model_ext1 = Classifier(dataset=dataset,estimator=ExtraTreesClassifier,parameters={'n_estimators':700,'max_depth':39,'n_jobs':-1,
							'criterion':'gini','min_samples_split':18},name='ext1')
model_ext2 = Classifier(dataset=dataset,estimator=ExtraTreesClassifier,parameters={'n_estimators':700,'max_depth':39,'n_jobs':-1,
							'criterion':'entropy','min_samples_split':18},name='ext2')


# Stack two models
# Returns new dataset with out-of-fold predictions
pipeline = ModelsPipeline(model_rf1,model_rf2,model_gdbt1,model_gdbt2,model_gdbt3,model_ext1,model_ext2)
stack_ds = pipeline.stack(k=5,seed=111)

# Train LinearRegression on stacked data (second stage)
stacker1 = Classifier(dataset=stack_ds, estimator=LogisticRegression,parameters={'C': 10})
# stacker2 = Classifier(dataset=stack_ds, estimator=LogisticRegression,parameters={'C': 1,'penalty':'l1'})
# stacker3 = Classifier(dataset=stack_ds, estimator=SVC,parameters={'probability':True,'C':100})
# stacker4 = Classifier(dataset=stack_ds, estimator=SVC,parameters={'probability':True,'C':10})
pre_y1 = stacker1.predict()
# pre_y2 = stacker2.predict()
# pre_y3 = stacker3.predict()
# pre_y4 = stacker4.predict()

#print(pre_y)
# 计算auc
# fpr, tpr, thresholds = metrics.roc_curve(test_y, pre_y1)
# auc=metrics.auc(fpr, tpr)
# print("AUC得分为：")
# print(auc)
# 计算auc
# fpr, tpr, thresholds = metrics.roc_curve(test_y, pre_y2)
# auc=metrics.auc(fpr, tpr)
# print("AUC得分为：")
# print(auc)
# 计算auc
# fpr, tpr, thresholds = metrics.roc_curve(test_y, pre_y3)
# auc=metrics.auc(fpr, tpr)
# print("AUC得分为：")
# print(auc)
# 计算auc
# fpr, tpr, thresholds = metrics.roc_curve(test_y, pre_y4)
# auc=metrics.auc(fpr, tpr)
# print("AUC得分为：")
# print(auc)

# # 设置参数
xgb = XGBClassifier(n_estimators = 1350,scale_pos_weight=4,nthread=20,seed=6,max_depth=3,min_child_weight=6,learning_rate=0.05,
							gamma=0,subsample=0.9,colsample_bytree=0.9,reg_alpha=8)
# 训练
xgb.fit(train_data, targets)
# 预测
pre_y = xgb.predict_proba(test_data)[:,1]
pre_y_categ = xgb.predict(test_data)

pre_y = 1/2*pre_y1 + 1/2*pre_y

evaluation_public = pd.DataFrame()
evaluation_public['EID'] = data_test['EID']
evaluation_public['FORTARGET']=pre_y_categ
evaluation_public['PROB'] = pre_y
print('success runing')
evaluation_public.to_csv('evaluation_public.csv',index=False)