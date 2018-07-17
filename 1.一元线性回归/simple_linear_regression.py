#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:38:56 2018

@author: lbb
"""

# 第一步 导入标准的库函数
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 第二步 构建数据集

dataset = pd.read_csv('Salary_Data.csv') #从文件中读取数据到dataset变量中

X = dataset.iloc[:,:-1].values #设置自变量
Y = dataset.iloc[:,1].values   #设定因变量


# 第三步 将数据分割成训练集和测试集
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0) #生成训练集和测试集


# 第四步 拟合线性回归模型
from sklearn.linear_model import LinearRegression #从python机器学习的库中导入线性回归函数
regressor = LinearRegression()
regressor.fit(X_train,Y_train) #用训练集中的数据，拟合模型


# 第五步 预测
y_pred = regressor.predict(X_test) #根据测试机通过模型进行预测

# 第六步 画图-训练集
plt.scatter(X_train,Y_train,color='red') #用测试集画点
plt.plot(X_train,regressor.predict(X_train),color='blue') #用测试集画线
plt.title('Salary VS Experience (traing set)') #设置标题
plt.xlabel('Years of Experience') #设置X坐标
plt.ylabel('Salary') #设置Y坐标
plt.show() #图表展示


# 画图 -测试集合
plt.scatter(X_test,Y_test,color='red') #用训练集画点
plt.plot(X_train,regressor.predict(X_train),color='blue') #模型
plt.title('Salary VS Experience (test set)') 
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
