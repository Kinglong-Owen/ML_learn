# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 09:21:46 2019

@author: Administrator
"""
import os
os.chdir(r'C:\Users\Administrator\Desktop')
import pandas as pd

dataset=pd.DataFrame({'house':['yes','no','no','yes','no','no','yes','no','no','no'],
                      'marry':['single','married','single','married','divorced','married','divorced','single','married','single'],
                      'income':[125,100,70,120,95,60,220,85,75,90],
                      'label':['no','no','no','no','yes','no','no','yes','no','yes']})

data_classic={'house':'scatter','marry':'scatter','income':'continuous','label':'scatter'}

def cal_Gini(dataset):
    item=set(dataset['label'])
    result=1
    length=len(dataset['label'])
    for i in item:
        result-=len(dataset[dataset['label']==i])**2/length**2
    return result

def find_best_split(dataset):
    best_split_gini=-10000
    best_split_feature=None
    best_split_value=0
    G=cal_Gini(dataset)
    for feature in dataset.columns[:-1]:
        if data_classic[feature]=='scatter':
            item=set(dataset[feature])        
            for i in item:
                nowgini=G
                sub_dataset=dataset[dataset[feature]==i]
                nowgini-=len(sub_dataset)/len(dataset)*cal_Gini(sub_dataset)
                
                sub_dataset=dataset[dataset[feature]!=i]
                nowgini-=len(sub_dataset)/len(dataset)*cal_Gini(sub_dataset)
                
                print(feature,',',i,',',nowgini)
                
                if best_split_gini<nowgini:
                    best_split_gini=nowgini
                    best_split_feature=feature
                    best_split_value=i  
        else:
            item=set(dataset[feature])
            item=sorted(item)
            item1=[round((item[i]+item[i+1])/2,4) for i in range(len(item)-1)]   #取中值
    
            for i in item1:         
                nowgini=G
                sub_dataset=dataset[dataset[feature]<i]    
                nowgini-=len(sub_dataset)/len(dataset)*cal_Gini(sub_dataset)  
          
                sub_dataset=dataset[dataset[feature]>=i]    
                nowgini-=len(sub_dataset)/len(dataset)*cal_Gini(sub_dataset)  
          
                print(feature,',',i,',',nowgini)
                
                if best_split_gini<nowgini:
                    best_split_gini=nowgini
                    best_split_feature=feature
                    best_split_value=i
                    
                print(nowgini,',',feature,',',i)
                
    return best_split_feature,best_split_value,best_split_gini

print(find_best_split(dataset))















