# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:58:36 2019

@author: Administrator
"""

import threading
import time
#def one():
#     for i in range(3):
#          print('第一个线程%d'%i)
#          time.sleep(1)
#
#def two():
#     for i in range(4):
#          print('第二个线程%d'%i)
#          time.sleep(1)
#
#if __name__=='__main__':
#     th1=threading.Thread(target=one)
#     th2=threading.Thread(target=two)
#     th1.start()
#     #th2.start()
#     print(th1.isAlive())
#     th1.join()
#     print(th1.isAlive())
#     th1.run()
#     print('线程1终止')

class mythread(threading.Thread):
     def __init__(self,n):
          super().__init__()
          self.n=n
     def run(self):
          for i in range(3):
               print('第%d个线程'%self.n)
               time.sleep(1)

mt1=mythread(1)
mt2=mythread(2)

mt1.start()
mt2.start()
     
     

     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     