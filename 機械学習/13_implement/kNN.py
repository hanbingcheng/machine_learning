# -*- coding: utf-8 -*-
'''
Created on 2018/01/20

@author: teikanhei
'''
import numpy as np
from numpy.random import randint,randn
import matplotlib.pyplot as plt
from sklearn import datasets

def euclideanDistance(x1, x2):
    x1 = np.asarray(x1).flatten()
    x2 = np.asarray(x2).flatten()
    
    return np.sqrt(np.sum(np.power((x1 - x2), 2)))

#Xr: Training data
#x: test data
#k: nearest k neighbor
# return class(y)
def kNN(dataset, x, k = 1):
    row = dataset.shape[0]
    Xdist = np.zeros((row, 1))
    for i in range(0,row):
        Xdist[i] = euclideanDistance(x, dataset[i, :-1])
    sortIdx = np.argsort(Xdist, axis = 0)
    topIdx = sortIdx[0:k].flatten()
    return dataset[topIdx]
    
    
# Editing TraingDatas
def editTrainData(dataset):
        nFold = 3  #样本随机划分为3个子集
        dataset_edit=dataset #当前样本集
        dataset_del = []
        #退出控制条件，若没有样本被剪辑掉，则退出
        while True:
            dataset_edit_pre = dataset_edit
            dataset_edit = np.array([])
            row = dataset_edit_pre.shape[0]
            #产生row1行1列的随机数，随机数的范围在0-s之间
            uu = randint(0, nFold,(row, 1))
           
            for i in range(0, nFold):
                data_edit= dataset_edit_pre[uu[:,0] == i] #test set   
                r = (i+1) % nFold #取余数
                data_train = dataset_edit_pre[uu[:,0] == r]  #reference set    
                row =  data_edit.shape[0]
                j = 0
                while j < row:
                    #用训练集中的样本对考试集中的样本进行最近邻分类
                    rClass = kNN(data_train, data_edit[j, :-1])[0, -1]
                    #如果类别不同，则从考试集中分类错误的样本去除
                    if rClass != data_edit[j, -1]: 
                        dataset_del.append(data_edit[j])
                        data_edit = np.delete(data_edit, j, axis =0)              
                        row = row-1
                    else:
                        j=j+1;
                if dataset_edit.size == 0:
                    dataset_edit = data_edit
                else:
                    dataset_edit = np.append(dataset_edit, data_edit, axis =0)
            if dataset_edit_pre.shape[0] == dataset_edit.shape[0]:
                break
        
        return (dataset_edit, np.array(dataset_del))

# 被删除样本周围m个样本中有不同类别的，则返回其m个样本
def boundaryDataSearch(dataset, dataset_del, k = 5):
   
    for i in range(dataset_del.shape[0]):
        neighbors = kNN(dataset, dataset_del[i, :-1], k)
        if len(np.unique(neighbors[:, -1])) > 1:
            return neighbors

# get index of data in dataset
def getIndexOfDataset(dataset, data):
    for i in range(0, dataset.shape[0]):
        if np.array_equal(dataset[i], data):
            return i
          
# Condense data
def condenseTrainData(dataset, dataset_del, startDataCnt = 5):
    # 得用删除的样本寻找出边界点
    neighbors = boundaryDataSearch(dataset, dataset_del, startDataCnt)
    
    for i in range(startDataCnt):
        idx = getIndexOfDataset(dataset, neighbors[i])
        data_store = np.array([dataset[idx,:]])
        ind = [i for i in range(dataset.shape[0]) if i != idx]
        data_grab = dataset[ind,:]
        
        while True:
            data_store_pre = data_store
            row = data_grab.shape[0]
            j = 0
            while j < row:
                gClass = kNN(data_store, data_grab[j,:-1])[0, -1]
                if gClass != data_grab[j, -1]:
                    data_store = np.append(data_store, [data_grab[j,:]], axis = 0)
                    data_grab = np.delete(data_grab, j, axis =0)
                    row = row-1
                else:
                    j = j + 1
            if data_store_pre.size == data_store.size or data_grab.size == 0:
                break
    return data_store

def draw(dataset, position, title):
    data0 = dataset[dataset[:, -1] == 0]
    data1 = dataset[dataset[:, -1] == 1]
    data2 = dataset[dataset[:, -1] == 2]
    plt.subplot(position)
    plt.plot(data0[:, 0], data0[:, 1], 'r^',
             data1[:, 0], data1[:, 1], 'bs',
             data2[:, 0], data2[:, 1], 'g*')
    plt.title(title)
       
if __name__ == "__main__":
    #use iris dataset
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    dataset = np.c_[iris_X, iris_y]
    np.random.shuffle(dataset)
    # 80% for train  and 20% for test
    train_data_cnt = int(dataset.shape[0] * 0.8)
    dataset_train = dataset[0: train_data_cnt, :]
    dataset_test = dataset[train_data_cnt:, :]
    # 初始样本分布图
    plt.figure("iris")
    plt.rcParams['font.sans-serif']=['SimHei']
    draw(dataset_train, 131, '初始样本分布图')

    # 剪辑样本
    dataset_train, dataset_del = editTrainData(dataset_train)
    # 剪辑后样本分布图
    draw(dataset_train, 132, '剪辑后样本分布图')
    
    # 压缩样本
    dataset_train = condenseTrainData(dataset_train, dataset_del, 5)
    # 压缩后样本分布图
    draw(dataset_train, 133, '压缩后样本分布图')
    plt.show()
    
    # test 
    ok = 0
    for i in range(dataset_test.shape[0]):
        prediction = kNN(dataset_test, dataset_test[i, :-1])[0, -1]
        if prediction == dataset_test[i, -1]:
            ok = ok + 1
    print ("precision is :{}".format(ok / dataset_test.shape[0]))