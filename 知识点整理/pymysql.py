# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:09:17 2019

@author: Administrator
"""
import os
os.chdir(r'C:\Users\Administrator\Desktop')
import pymysql


conn=pymysql.connect(host='127.0.0.1', user='root', password='1', database='test', 
port=3306, charset='utf8')
cur=conn.cursor()
print("*****====================================*****")
# 使用 execute()方法执行 SQL 查询
# 使用 fetchone() 方法获取单条数据.

cur.execute('select * from employees;')
data=cur.fetchall()
print(data)
print("*****====================================*****")
