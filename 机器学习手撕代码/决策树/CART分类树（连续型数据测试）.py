# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:59:36 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
#data=pd.DataFrame([[0.697,0.994,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719],
#                   [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]]).T
#data.columns=['midu','label']

data=pd.DataFrame({'house':['yes','no','no','yes','no','no','yes','no','no','no'],
                      'marry':['single','married','single','married','divorced','married','divorced','single','married','single'],
                      'income':[125,100,70,120,95,60,220,85,75,90],
                      'label':['no','no','no','no','yes','no','no','yes','no','yes']})

def cal_Gini(dataset):
    item=set(dataset['label'])
    result=1
    length=len(dataset['label'])
    for i in item:
        result-=len(dataset[dataset['label']==i])**2/length**2
    return result

def find_best_split(dataset):
    best_split_gini=10000
    best_split_feature=None
    best_split_value=0
    for feature in dataset.columns[:-1]:
        feature='income'        
        item=set(dataset[feature])
        item=sorted(item)
        G=cal_Gini(dataset)
        item1=[round((item[i]+item[i+1])/2,4) for i in range(len(item)-1)]   #取中值

        for i in item1:         
            nowgini=G
            sub_dataset=dataset[dataset[feature]<i]
            gini=cal_Gini(sub_dataset)         
            nowgini-=len(sub_dataset)/len(dataset)*gini
      
            sub_dataset=dataset[dataset[feature]>=i]
            gini=cal_Gini(sub_dataset)         
            nowgini-=len(sub_dataset)/len(dataset)*gini
      
            if best_split_gini>nowgini:
                best_split_gini=nowgini
                best_split_feature=feature
                best_split_value=i
                
            print(nowgini,',',feature,',',i)

     
        break

find_best_split(data)
    
