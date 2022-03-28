import numpy as np
import math
import random


def loadData(path):
    '''

    :param path: 数据路径
    :return: 返回数据集及数据集标签
    '''
    dataArr=[];labelArr=[]
    fr=open(path)
    for line in fr.readlines():
        curLine=line.strip().split(',') #curLine[0]为标记信息
        dataArr.append([int(num)/255 for num in curLine[1:]])
        if int(curLine[0])==0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    return dataArr,labelArr

class SVM:
    def __init__(self,trainDataList,trainLabelList,sigma=10,C=200,toler=0.001):
        '''

        :param trainDataList:
        :param trainLabelList:
        :param sigma: 高斯核参数
        :param C: 惩罚参数
        :param toler:松弛变量
        '''
        self.trainDataMat=np.mat(trainDataList)
        self.trainLabelMat=np.mat(trainLabelList).T
        self.m,self.n=np.shape(self.trainDataMat)
        self.sigma=sigma
        self.C=C
        self.toler=toler

        self.k=self.calcKernel()  #采用高斯核函数 初始化
        self.b=0  # bias
        self.alpha=[0]*self.trainDataMat.shape[0] #拉格朗日系数
        self.E=[0*self.trainLabelMat[i,0] for i in range(self.trainLabelMat.shape[0])]
        # smo算法中的Ei 即输入xi的预测值与真实yi之间的差
        self.supportVecIndex=[]

    def  calcKernel(self):
        '''
        高斯核函数
        :return: 高斯核矩阵 m*m阶方阵 m为训练集数
        '''
        k=[[0 for i in range(self.m)] for j in range(self.m)]

        #遍历Xi
        for i in range(self.m):
            if i%100==0:
                print('构造核:',i,self.m)
            X=self.trainDataMat[i,:]

            #遍历Xj
            for j in range(i,self.m):
                Z=self.trainDataMat[j,:]
                result=(X-Z)*(X-Z).T
                result=np.exp(-1*result/(2*self.sigma)**2)
                k[i][j]=result
                k[j][i]=result  #正定矩阵
        return k

    def isSatisfyKKT(self,i):
        '''

        :param i: 第i个alpha是否满足KKT条件
        :return: True or False
        '''
        gxi=self.calc_gxi(i)
        yi=self.trainLabelMat[i]

        #选择第一个alpha 7.111~~~7.114 p149

        if (math.fabs(self.alpha[i])<self.toler) and (yi*gxi>=1):
            return True
        elif (math.fabs(self.alpha[i]-self.C)<self.toler) and (yi*gxi<=1):
            return True
        elif (self.alpha[i]>self.toler) and (self.alpha[i]<(self.C+self.toler)) and (math.fabs(yi*gxi-1)<self.toler):
            return True

        return False

    def calc_gxi(self,i):
        '''

        :param i: x的下标
        :return: g(xi）  7.104
        '''
        gxi=0
        # 因为g(xi)是一个求和式+b的形式，普通做法应该是直接求出求和式中的每一项再相加即可
        # 但是读者应该有发现，在“7.2.3 支持向量”开头第一句话有说到“对应于α>0的样本点
        # (xi, yi)的实例xi称为支持向量”。也就是说只有支持向量的α是大于0的，在求和式内的
        # 对应的αi*yi*K(xi, xj)不为0，非支持向量的αi*yi*K(xi, xj)必为0，也就不需要参与
        # 到计算中。也就是说，在g(xi)内部求和式的运算中，只需要计算α>0的部分，其余部分可
        # 忽略。因为支持向量的数量是比较少的，这样可以再很大程度上节约时间
        # 从另一角度看，抛掉支持向量的概念，如果α为0，αi*yi*K(xi, xj)本身也必为0，从数学
        # 角度上将也可以扔掉不算
        # index获得非零α的下标，并做成列表形式方便后续遍历
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        # 遍历每一个非零α，i为非零α的下标
        for j in index:
            gxi+=self.alpha[j]*self.trainLabelMat[j]*self.k[j][i]
            gxi+=self.b             #7.104式

        return gxi
    def calc_Ei(self,i):
        '''
        计算 Ei 看7.105
        :param i: Ei的下标
        :return: Ei
        '''
        gxi=self.calc_gxi(i)
        return gxi-self.trainLabelMat[i]

    def getAlphaJ(self,E1,i):
        '''
        选择som算法的第二个参数
        :param Ei:
        :param i: 第一个alpha的下标
        :return:
        '''
        E2=0
        #初始化|E1-E2|
        maxE1_E2=-1
        #初始话第二个变量的下标
        maxIndex=-1
        # 获得Ei非0的对应索引组成的列表，列表内容为非0Ei的下标i
        nozeroE=[i for i,Ei in enumerate(self.E) if Ei !=0]
        for j in nozeroE:
            E2_tmp=self.calc_Ei(j)
            if math.fabs(E1-E2_tmp)>maxE1_E2:
                #更新
                maxE1_E2=math.fabs(E1-E2_tmp)
                E2=E2_tmp
                maxIndex=j
        if maxIndex==-1:
            maxIndex=i
            while maxIndex==i:
                maxIndex=int(random.uniform(0,self.m))
            E2=self.calc_Ei(maxIndex)
        return E2,maxIndex

    def train(self,iter=100):
        '''

        :param iter: 迭代次数
        :return:
        '''
        iterStep=0;parameterChanged=1 #迭代次数，参数改变量
        # parameterChanged==0时表示上次迭代没有参数改变，如果遍历了一遍都没有参数改变，说明
        # 达到了收敛状态，可以停止了
        while(iterStep<iter) and (parameterChanged>0):
            print('iter:%d:%d'%(iterStep,iter))
            iterStep+=1
            parameterChanged=0
            #第一层循环，找smo算法的第一个参数
            for i in range(self.m):
                if self.isSatisfyKKT(i)==False: #不满足KKT条件，则寻找第二个参数
                    E1=self.calc_Ei(i)
                    E2,j=self.getAlphaJ(E1,i)
                    y1=self.trainLabelMat[i]
                    y2=self.trainLabelMat[j]

                    alphaOld_1=self.alpha[i]
                    alphaOld_2=self.alpha[j]
                    # p144
                    if y1!=y2:
                        L=max(0,alphaOld_2-alphaOld_1)
                        H=min(self.C,self.C+alphaOld_2-alphaOld_1)
                    else:
                        L=max(0,alphaOld_2+alphaOld_1-self.C)
                        H=min(self.C,alphaOld_2+alphaOld_1)

                    if L==H: #无法继续优化，跳入下一次循环
                        continue
                    K11=self.k[i][i]
                    K22=self.k[j][j]
                    K12=self.k[i][j]
                    K21=self.k[j][i]
                    ET=K11+K22-2*K12
                    alphaNew_2=alphaOld_2+y2*(E1-E2)/ET
                    if alphaNew_2<L:
                        alphaNew_2=L
                    elif alphaNew_2>H:
                        alphaNew_2=H

                    alphaNew_1=alphaOld_1+y1*y2*(alphaOld_2-alphaNew_2)
                    bNew_1=-1*E1-y1*K11*(alphaNew_1-alphaOld_1)-y2*K21*(alphaNew_2-alphaOld_2)+self.b
                    bNew_2=-1*E2-y1*K12*(alphaNew_1-alphaOld_1)-y2*K22*(alphaNew_2-alphaOld_2)+self.b

                    if (alphaNew_1>0) and (alphaNew_1<self.C):
                        bNew=bNew_1
                    elif (alphaNew_2>0) and (alphaNew_1<self.C):
                        bNew=bNew_2
                    else:
                        bNew=(bNew_1+bNew_2)/2

                    #存储更新值
                    self.alpha[i]=alphaNew_1
                    self.alpha[j]=alphaNew_2
                    self.b=bNew
                    self.E[i]=self.calc_Ei(i)
                    self.E[j]=self.calc_Ei(j)

                    # 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    # 反之则自增1
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1

                    # 打印迭代轮数，i值，该迭代轮数修改α数目
                print("iter: %d i:%d, pairs changed %d" % (iterStep, i, parameterChanged))

                # 全部计算结束后，重新遍历一遍α，查找里面的支持向量
                for i in range(self.m):
                    # 如果α>0，说明是支持向量
                    if self.alpha[i] > 0:
                        self.supportVecIndex.append(i)

    def gusskernel(self,X1,X2):
        '''

        :param X1: 特征向量
        :param X2: 特征向量
        :return: 高斯核函数
        '''
        result=(X1-X2)*(X1-X2).T
        K=np.exp(-1*result/(2*self.sigma**2))
        return K

    def predict(self, x):
        '''
        对样本的标签进行预测
        公式依据非线性支持向量分类机”中的式7.94
        :param x: 要预测的样本x
        :return: 预测结果
        '''

        result = 0
        for i in self.supportVecIndex:
            # 遍历所有支持向量，计算求和式
            tmp = self.gusskernel(self.trainDataMat[i, :], np.mat(x))
            # 对每一项子式进行求和，最终计算得到求和项的值
            result += self.alpha[i] * self.trainLabelMat[i] * tmp
        result += self.b
        # 使用sign函数返回预测结果
        return np.sign(result)


    def test(self,dataList,labelList):
        '''

        :param dataList: 输入特征
        :param labelList: 正确标签
        :return: 返回正确率
        '''
        error=0
        for i in range(len(dataList)):
            predictLabel=self.predict(dataList[i])
            if predictLabel!=labelList[i]:
                error+=1
        return 1-error/len(dataList)



if __name__=='__main__':
    #获取训练集
    trainPath='F:\python_code\Pycharm_\ML\SVM\data\mnist_train\mnist_train.csv'
    testPath='F:\python_code\Pycharm_\ML\SVM\data\mnist_test\mnist_test.csv'
    trainDataList,trainLabelList=loadData(trainPath)
    testDataList,testLabelList=loadData(testPath)
    # print(trainLabelList)
    svm = SVM(trainDataList[:100], trainLabelList[:200], 10, 200, 0.001)
    svm.train()
    accuracy = svm.test(testDataList[:100], testLabelList[:100])
    print('真确率:%.2f'% (accuracy * 100),'%')

