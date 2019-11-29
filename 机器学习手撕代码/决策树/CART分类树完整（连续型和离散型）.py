# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:38:34 2019

@author: Administrator
"""
import math
import pandas as pd
import numpy as np

dataset=pd.DataFrame({'house':['yes','no','no','yes','no','no','yes','no','no','no'],
                      'marry':['single','married','single','married','divorced','married','divorced','single','married','single'],
                      'income':[125,100,70,120,95,60,220,85,75,90],
                      'label':['no','no','no','no','yes','no','no','yes','no','yes']})

data_classic={'house':'scatter','marry':'scatter','income':'continuous','label':'scatter'}
#def get_info(data1):
#     cl_list=set(data1[-1])       #统计种类
#     p={}
#     for i in cl_list:        
#          num=data1[-1].value_counts()[i]#计算每个种类的数目
#          p[i]=num/len(data1)           #构造字典
#     tem=-sum([x*math.log2(x) for x in p.values()])#计算结果
#     return [p,tem]

#计算香农熵，计算都是标签列的香农熵，新增属性后，计算新属性的每一类别的香农熵
#公式见markdown
#def get_info1(data1):
#     n=data1.shape[0]
#     iset=data1.iloc[:,-1].value_counts()#得到一个series，索引是每个类别的名字，数值每个类别的数量
#     p=iset/n                           #numpy会自动广播
#     ent=(-p*np.log2(p)).sum()
#     return ent
#
##计算信息增益
#def get_gain(A):
#     info_d=get_info1(data)   #先计算总体数据的香农熵
#     item=set(data[A])
#     info_a_D=0
#     
#     for i in item:           #计算新属性的每个类别的香农熵
#          ch_info=get_info1(data[data[A]==i])
#          info_a_D+=(len(data[data[A]==i])/len(data))*ch_info   #新属性每个类别占新属性的比例*每个类别的香浓熵
#         
#     return info_d-info_a_D   #相减得到增益

#计算最佳分隔，返回最佳分隔的列名  
#def bestsplit(dataset):
#     bestgian=0
#     for index,featname in enumerate(dataset.columns[:-1]):  #每个属性都计算信息增益
#          info_gain=get_gain(featname)
#          
#          if info_gain>bestgian:  #找到信息增益最大的属性名
#               bestgian=info_gain
#               axis=index
#     return axis

#返回axis列中，按value值筛选出来的数据集，并且删删掉axis列
def mysplit(dataset,feature,value,d_c,key):
    if d_c=='scatter':
        if key=='left':        
            redataset=dataset[dataset[feature]==value].drop(feature,axis=1)
        else:
            redataset=dataset[dataset[feature]!=value].drop(feature,axis=1)
    else:
        if key=='left':
            redataset=dataset[dataset[feature]<=value]
        else:
            redataset=dataset[dataset[feature]>value]
    return redataset
     

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
                
               # print(feature,',',i,',',nowgini)
                
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
          
              # print(feature,',',i,',',nowgini)
                
                if best_split_gini<nowgini:
                    best_split_gini=nowgini
                    best_split_feature=feature
                    best_split_value=i
                    
                #print(nowgini,',',feature,',',i)
                
    return best_split_feature,best_split_value
     
#递归构建决策树，把决策树保存在字典里面   
def createTree(dataset):                               #传入上一级被切分好的数据集
    #featlist=list(dataset.columns)                    #返回一个列表，属性名的列表
    classlist=dataset.iloc[:,-1].value_counts()       #计算这个数据集的标签的每个类别的数量   
    #print(dataset)
    if classlist[0]==len(dataset) or len(dataset)==1:  #如果全是1或者全是0，或者只剩下一行数据了，那就返回这个标签
        print(dataset)
        return [classlist.index[0],len(dataset)]
      
    feature,value=find_best_split(dataset)                           #计算最佳分裂的
                                                                      #最佳分类属性名
    mytree={feature:{},'value':value}                              #设置当前树的根节点的名
    
                                                           #计算当前数据集的最佳分裂属性的类别数
                                                               #按每个类别生成下一级的各个节点，有多少个类别就生成多少个子节点
                                                               #把dataset[dataset[bestfeat]==balue]带进去递归
                                                               #把递归的结果保存到根节点的value值中
     
    mytree[feature]['left']=createTree(mysplit(dataset,feature,value,data_classic[feature],'left'))  
    mytree[feature]['right']=createTree(mysplit(dataset,feature,value,data_classic[feature],'right'))  
     
     
    return mytree#返回这颗树的根节点

mt=createTree(dataset)

print(mt)
















