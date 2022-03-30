import numpy as np
import matplotlib.pyplot as plt
import operator
import os

class KNN:
    def __init__(self,trainData,trainLabels,testData,testLabels,Linear=True):
        '''

        :param trainData:
        :param trainLabels:
        :param train_size:
        :param valid_size:
        :param test_size:
        '''
        self.trainData=trainData
        self.trainLabels=trainLabels
        self.m=np.shape(trainData)[0]
        self.testData=testData
        self.testLabels=testLabels
        self.n=np.shape(testData)[0]
        self.trainSet=self.normlize(self.trainData)
        self.testSet=self.normlize(self.testData)
        self.Linear=Linear

    def normlize(self,inputs,Linear=False):
        X_normal=None
        if Linear:
            maxSet=np.amax(inputs,axis=0)
            minSet=np.amin(inputs,axis=0)
            X_normal=(inputs-minSet)/(maxSet-minSet)
        else:
            X_normal=inputs
        return X_normal

    def cala_dist(self,trainSet,x):
        error=(x-trainSet)**2
        dist=np.sum(error,axis=1)**0.5
        return dist

    def fit(self,x,train_set,train_label,k=3):
        dist=self.cala_dist(train_set,x)
        sortedIndex=np.argsort(dist)
        countList={}
        for i in range(k):
            countList[train_label[sortedIndex[i]]]=countList.get(train_label[sortedIndex[i]],0)+1
        sortedDict=sorted(countList.items(),key=operator.itemgetter(1),reverse=True)
        return sortedDict[0][0]

    def find_k(self):
        ks=np.arange(1,50)
        dataSet=self.normlize(self.trainData)
        m=int(self.m*0.75)
        train_set,train_label=dataSet[:m,:],self.trainLabels[:m]
        valid_set,valid_label=dataSet[m:,:],self.trainLabels[m:]
        errorList=[]
        for k in ks:
            errorCount=0
            for i in range(int(m*0.25)):
                result=self.fit(valid_set[i,:],train_set,train_label,k)
                if result!=valid_label[i]:
                    errorCount+=1
            errorCount/=(m*0.25)
            errorList.append(errorCount)
        fig=plt.figure(dpi=600)
        plt.plot(ks,errorList,c='red')
        plt.xlabel('k')
        plt.ylabel('error_rate')
        plt.grid()
        plt.show()
        return np.argsort(np.array(errorList),kind='stable')[0]+1,errorList #kind 为排序算法

    def test(self):
        test_set,test_label=self.testSet,self.testLabels
        train_set,train_label=self.trainSet,self.trainLabels
        errorCount=0
        test_size=self.n
        for i in range(test_size):
            result=self.fit(test_set[i,:],train_set,train_label,k=5)
            if result!=test_label[i]:
                errorCount+=1
        error_rate=errorCount/test_size
        return 1-error_rate

def loadData(path):
    dataSet=[];label=[]
    fr=open(path)
    for line in fr.readlines():
        line=line.strip().split('\t')
        dataSet.append(list(map(float,line[:-1])))
        label.append(line[-1])
    return np.array(dataSet),label

#手写数字识别

def img_to_vector(filename):
    returnVector=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        line=fr.readline()
        for j in range(32):
            returnVector[0,32*i+j]=int(line[j])
    return returnVector

def getData():
    train_labels=[]
    trainFileList=os.listdir('F:/python_code/Pycharm_/ML/KNN/data/digits/trainingDigits')
    m=len(trainFileList)
    train_data = np.zeros((m,1024))
    for i in range(m):
        fileName=trainFileList[i]
        fileStr=fileName.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        train_labels.append(classNumStr)
        train_data[i,:]=img_to_vector('F:/python_code/Pycharm_/ML/KNN/data/digits/trainingDigits/%s'%fileName)
    test_labels=[]
    testFileList=os.listdir('F:/python_code/Pycharm_/ML/KNN/data/digits/testDigits')
    m = len(testFileList)
    test_data =np.zeros((m,1024))
    for i in range(m):
        fileName = testFileList[i]
        fileStr = fileName.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        test_labels.append(classNumStr)
        test_data[i, :] = img_to_vector('F:/python_code/Pycharm_/ML/KNN/data/digits/testDigits/%s' % fileName)
    return train_data,train_labels,test_data,test_labels

if __name__=='__main__':
    path='F:\python_code\Pycharm_\ML\KNN\data\datingTestSet.txt'
    dataSet,label=loadData(path)
    knn=KNN(dataSet,label,dataSet,label)
    K,errorList=knn.find_k()
    rate=knn.test()
    print('准确率为:{:0.3f}%'.format(rate*100))
    # train_data, train_labels, test_data, test_labels=getData()
    # knn=KNN(train_data,train_labels,test_data,test_labels,Linear=False)
    # K, errorList = knn.find_k()
    # rate = knn.test()
    # print('准确率为:{:0.3f}%'.format(rate * 100))