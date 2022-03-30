import numpy as np

def calcDist(x1,x2,Euc=True):
    '''
    计算距离/欧式距离，or曼哈顿距离
    :param x1: 向量1
    :param x2: 向量2
    :return:
    '''
    if Euc==True:
        return np.sqrt(np.sum(np.square(x1-x2)))
    else:
        return np.sum(np.abs(x1-x2))

def getBestClose(trainDataMat,trainLabelMat,x,k):
    '''
    取x对训练样本最近的值
    :param trainDataMat: 训练数据
    :param trainLabelMat: 训练标签
    :param x: 测试点
    :param k: 最近样本数量
    :return: 预测分类
    '''

    distList=[0]*len(trainLabelMat) #存储测试点到各样本点的距离
    for i in range(len(trainDataMat)):
        x1=trainDataMat[i]
        x2=x
        dist=calcDist(x1,x2)
        distList[i]=dist

    #排序找到最近的点
    #按值由小到大，按索引排序
    #即最前面的索引所对应的值道测试点的距离最小
    topList=np.argsort(np.array(distList))[:k] #取前k个值

    labelList=[0]*10
    for index in topList:
        labelList[int(trainLabelMat[index])]+=1
    return labelList.index((max(labelList)))

def test(trainDataArr,trainLabelArr,testDataArr,testLabelArr,k=3,testcount=200):
    '''

    :param trainDataArr: 训练数据集
    :param trainLabelArr: 训练标签集
    :param testDataArr: 测试数据集
    :param testLabelArr: 测试标签集
    :param k:
    :return: 分类正确率
    '''
    trainDataMat=np.mat(trainDataArr);trainLabelMat=np.mat(trainLabelArr).T
    testDataMat=np.mat(testDataArr);testLabelMat=np.mat(testLabelArr).T
    errorCount=0

    for i in range(testcount):
        x=testDataMat[i]
        y=getBestClose(trainDataMat,trainLabelMat,x,k)
        if y!=testLabelMat[i]:
            errorCount+=1
    return 1-(errorCount/testcount)

def loadData(path):
    dataList=[];labelList=[]
    fr=open(path)
    for line in fr.readlines():
        curLine=line.strip().split(',')
        dataList.append([int(num) for num in curLine[1:]])
        labelList.append(int(curLine[0]))
    return dataList,labelList

if __name__=='__main__':
    trainPath='F:\python_code\Pycharm_\ML\KNN\data\mnist_train.csv'
    testpath='F:\python_code\Pycharm_\ML\KNN\data\mnist_test.csv'

    trainDataList,trainLabelList=loadData(trainPath)
    testDataList,testLabelList=loadData(testpath)

    accur=test(trainDataList,trainLabelList,testDataList,testLabelList,25)
    print(accur)



