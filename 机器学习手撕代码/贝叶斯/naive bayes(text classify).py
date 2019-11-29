# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:41:54 2019

@author: Administrator
"""
#Naive Bayes Classifilter

import numpy as np
def load_data_set():
    """
    创建数据集,都是假的 fake data set 
    :return: 单词列表posting_list, 所属类别class_vec
    """
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is 侮辱性的文字, 0 is not
    return posting_list, class_vec

def create_vocab_list(data_set):
     result=set()
     for post in data_set:
          tem=set(post)
          result=tem|result
     return list(result)

def set_of_words2vec(word_vec, input_set):
     #属于一个文本，构造该文本的词向量，返回
     result=[0]*len(word_vec)
     for word in input_set:
          if word in word_vec:
               result[word_vec.index(word)]=1
     return result


def _train_naive_bayes(train_mat, train_category):
     ''' 
     train_mat 是已经转换成词向量的输入数据，size=源文本条数*词向量长度
     计算P(w|ci)   每个类别下面每个单词出现的概率，等于该类别下该单词出现的次数/该类别所有单词数
     返回值，两个个跟词向量一样长度的列表，每个元素代表该类别下该单词出现的概率;
     计算P(ci) 标签为1的概率，数值型
     '''
     global word_vec
     P_label_1=np.ones(len(word_vec))
     P_label_0=np.ones(len(word_vec))
     sum_1=2                              #标签为1的所有单词数
     sum_0=2
     for index,label in enumerate(train_category):
          if label==0:
               sum_0+=sum(train_mat[index])
               P_label_0+=train_mat[index]
          else:
               sum_1+=sum(train_mat[index])
               P_label_1+=train_mat[index]
     P_label_0=np.log(P_label_0/sum_0)
     P_label_1=np.log(P_label_1/sum_1)
     
     P_C_1=sum(train_category)/len(train_category)
     return P_label_1,P_label_0,P_C_1      
     
def classify_naive_bayes(vec2classify, p0vec, p1vec, p_class1):      
     p_1=np.sum(vec2classify*p1vec)+np.log(p_class1)
     p_0=np.sum(vec2classify*p0vec)+np.log(1-p_class1)
     print(p_1)
     if p_1>p_0:
          return 1
     else:
          return 0
               


if __name__=='__main__':
     list_post,label_list=load_data_set()
     word_vec=create_vocab_list(list_post)
     
     text_vector=[]
     for post in list_post:
          text_vector.append(set_of_words2vec(word_vec,post))


     P_label_1,P_label_0,P_C_1=_train_naive_bayes(text_vector, label_list)
#     for i in range(len(word_vec)):
#          print(word_vec[i],' ',P_label_1[i],' ' ,P_label_0[i])

     test_one = ['love', 'my', 'dalmation']
     test_one_vec=set_of_words2vec(word_vec,test_one)
     classify_result=classify_naive_bayes(test_one_vec, P_label_0, P_label_1, P_C_1)
     print(classify_result)
     
     test_two = ['stupid', 'garbage']
     test_two_vec=set_of_words2vec(word_vec,test_two)
     classify_result=classify_naive_bayes(test_two_vec, P_label_0, P_label_1, P_C_1)
     print(classify_result)


