# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:23:09 2019

@author: Administrator
"""

#iris

import numpy as np
import pandas as pd
import random
import  sklearn.datasets as skd
iris=skd.load_iris()

dataSet=pd.DataFrame(iris['data'],columns=iris['feature_names'])
dataSet['target']=iris['target']
dataSet['target']=dataSet['target'].astype(str)
dataSet['target'].replace(to_replace=['0','1','2'],value=['setosa','versicolor','virginica'],inplace=True)


def randSplit(dataSet,rate):
     l=list(dataSet.index)
     random.shuffle(l)
     dataSet.index=l
     n=dataSet.shape[0]
     m=int(n*rate)
     train=dataSet.loc[range(m),:]
     test=dataSet.loc[range(m,n),:]
     dataSet.index=range(dataSet.shape[0])
     test.index=range(test.shape[0])
     return train,test

def gnb_classify(train,test):
     global labels
     labels=iris['target_names']
     _mean=[]
     _stds=[]
     for l in labels:
          item=train[train['target']==l]
          _mean.append(item.mean())
          _stds.append(item.std())
     _mean=pd.DataFrame(_mean)
     _stds=pd.DataFrame(_stds)
     #_mean.index=labels
    # _stds.index=labels
     return _mean,_stds

def test_classify(test):
     global mesa,stds,labels
     result=[]
     for j in range(test.shape[0]):
          iset=test.iloc[j,:-1].tolist()
          
          '''
          iprob的size=类别数*特征数
          表示这个样本的各个特征在高斯分布中的概率，即P(特征|类别i)，所以第i行累积乘就是P(类别i|特征)
          
          特征1            特征2         特征3          特征3         
          P(特征1|类别1)   P(特征2|类别1) P(特征3|类别1) P(特征4|类别1)
          ...
          
          '''
          iprob=np.exp(-(iset-mean)**2/(2*stds))/(np.sqrt(2*np.pi*stds)) 
          prob=1
          for k in range(test.shape[1]-1):
               prob*=iprob[iprob.columns[k]]    
          print(prob)     
          result.append(labels[np.argmax(prob)])
     test['predict']=result
     acc=(test.iloc[:,-1]==test.iloc[:,-2]).mean()
    # print(test['predict'])
     print(acc)
     
labels=iris['target_names']
train_data,test_data=randSplit(dataSet,0.8)
mean,stds=gnb_classify(train_data,test_data)

test_classify(test_data)















