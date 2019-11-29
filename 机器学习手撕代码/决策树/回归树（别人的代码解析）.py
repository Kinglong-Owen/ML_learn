# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:35:16 2019

@author: Administrator
"""
import os
import numpy as np
os.chdir(r'C:\Users\Administrator\Desktop')
import pandas as pd
def createTree(dataSet, leafType, errType, cond=(1,4)):
    ''' 创建回归树/模型树。
    :param dataSet:
    :param leafType:
    :param errType:
    :param cond: 预剪枝条件
    :return:
    输入：要被划分的数据集，？，？，预剪枝的条件
    功能:给传入的数据寻找最佳分裂并且划分成左右子树，保存在retTree字典中
    返回值：retTree字典
    '''
    #调用choosetbestsplit函数寻找最佳分裂，传入数据集，
    feature, value = chooseBestSplit(dataSet, leafType, errType, cond)
    #如果返回特征是空，就作为叶子节点，只保存值
    if feature == None:
        return value
    
    retTree = {}
    retTree['spInd'] = feature    #split index
    retTree['spVal'] = value      #split value
    
    #寻找最佳分裂，并且返回左右子树的数据集
    lSet,rSet = binSplitDataSet(dataSet, feature, value)  
    
    #在字典里保存左右子树，并对左右子树进行递归划分
    retTree['left'] = createTree(lSet, leafType, errType, cond)
    retTree['right'] = createTree(rSet, leafType, errType, cond)
    return retTree

def binSplitDataSet(dataSet, feature, value):
    ''' 根据属性feature的特定value划分数据集        
    :param dataSet:
    :param feature:
    :param value:
    :return:
    输入：要被划分的数据集，最佳分裂的特征，最佳分裂的特征的值
    功能：把数据集按feature特征中的value值划分为左右子树
    返回值：左右子树的数据集
    '''
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1


def chooseBestSplit(dataSet, leafType, errType, cond=(1,4)):
    ''' 选择最佳待切分feature以及value
    
    :param dataSet:
    :param leafType:叶子节点的构建方法，是一个函数
    :param errType: 总均方差计算方法，是一个函数
    :param cond:
    :return:
    输入：数据集，叶子节点的构造方法，总均方误差的计算方法，
    功能：寻找最佳分裂。寻找过程：循环每个特征的每个值，并计算每个特征每个值的回归误差，找到最小的回归误差，记录该特征和值
    返回值：
    '''
    tolS = cond[0]
    tolN = cond[1]
  

    # dataSet中都属于同一类别，返回空的特征值和一个叶子节点值
    if len(set(dataSet['label'])) == 1:
        return None, leafType(dataSet)

    m,n = np.shape(dataSet)                                             #数据集的size
    S = errType(dataSet)
    bestS = np.inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A[0]):
            mat0,mat1 = binSplitDataSet(dataSet, featIndex, splitVal)   #循环寻找每个特征的每个值划分后的左右子树
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):      #划分之后个数小于tolN就结束舍弃这个数值
                continue;
            newS = errType(mat0) + errType(mat1)                        #计算左右子树的回归误差
            if newS < bestS:                                            #如果误差小于bestS,替换
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果当前的数据集的回归误差和最佳分裂的回归误差差值足够小，就没必要再分裂下去了，停止分裂，
    #并返回空和一个叶子节点，否则正常返回最佳分裂的特征名和特征值          
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    return bestIndex, bestValue    

#对于回归树的叶子节点的计算
#返回数据集的均值作为回归树的叶子节点的数据
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])#均值

#对于回归树的回归误差的计算
#返回一个数据集的总方差，用于计算
def regErr(dataSet):
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]


##对于模型树
#def linearSolve(dataSet):
#    ''' 对dataSet进行线性拟合
#
#    :param dataSet: 
#    :return: 
#    '''
#    m,n = np.shape(dataSet)
#    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))
#    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:, -1]
#    xTx = X.T * X
#    if np.linalg.det(xTx) == 0.0:
#        raise NameError('This matrix is singular, cannot do inverse,\n\
#        try increasing the second value of ops')
#    ws = xTx.I * X.T * Y
#    return ws,X,Y
#
#
#def modelLeaf(dataSet):
#    ws,X,Y = linearSolve(dataSet)
#    return ws
#
#
#def modelErr(dataSet):
#    ws, X, Y = linearSolve(dataSet)
#    yHat = X * ws
#    return sum(np.power(Y - yHat, 2))

X_data_raw = np.linspace(-3, 3, 50)
np.random.shuffle(X_data_raw)
y_data = np.sin(X_data_raw)
X_data = np.transpose([X_data_raw])
Y_data = y_data + 0.1 * np.random.randn(y_data.shape[0])

data=pd.DataFrame([X_data,Y_data]).T
data.columns=['x','label']


createTree(data, regLeaf, regErr, (1,4))
