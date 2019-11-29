from numpy import *
from time import sleep
# 依旧是数据的准备
def loadDataSet(fileName):
    # 定义保存样本和标签的列表;
    dataMat = []; labelMat = [a]
    # 打开数据文件
    fr = open(fileName)
    
    # 读取文件中的每一行;
    for line in fr.readlines():
        # 将每行的数据通过制表符分开;
        lineArr = line.strip().split('\t')
        # 前两个为样本数据数据，第二个为标签数据;
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    # 返回样本和标签，用于SVM的训练;
    return dataMat,labelMat
# 用于产生一个随机数，用于下面随机获取一个样本;
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j
# 相当于给定alpha一个范围，大于最大值的话，赋值为最大值，小于最小值的话，就赋值最小值; 
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

# 简化版的smo算法求alpha和b;
# 下面是smo算法的流程;
# 此简化版的smo是严格按照上一节MachineLN之SVM（2）的手撕smo来的；
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 将样本集转化为矩阵格式， 将样本的标签也转化为矩阵格式和，用于后面的矩阵运算， 主要两个地方用到：预测和eta。 
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    # 初始化偏置，然后获取矩阵的行数和列数， 行数用来输出初始化下面的alpha. 
    b = 0; m,n = shape(dataMatrix)
    # 初始化alphas为为m行1列；
    alphas = mat(zeros((m,1)))
    iter = 0
    # 定义迭代次数
    while (iter < maxIter):
        alphaPairsChanged = 0
        # 遍历每个样本;
        for i in range(m):
            # 计算第i个样本的预测标签; 用于计算差值； 
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # 计算两个的差值用于 KKT 条件的判断
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            # 正间隔 和 负间隔 都会被测试; 并且还要保证 alpha的值在 [0, C]之间
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 从 i到m中随机选择一个样本
                j = selectJrand(i,m)
                # 计算此样本的预测值
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                # 预测值和真实值的差值： 用于后面计算alpha. 
                Ej = fXj - float(labelMat[j])
                # 用于保存未更新的alpha，方便b的计算;
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                # 计算不同的情况下 aphpa 的最小值和最大值， 这里可以参考手撕smo;
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                # 下面就是计算alpha2 和 进行剪枝后，求alpha1;
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                # 更新参数b1， b2， 和手撕smo算法流程一样;
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                # 根据参数b1， b2得到b;
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas
