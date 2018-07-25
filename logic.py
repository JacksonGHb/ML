# _*_ coding : utf-8 _*_
# Time : 2018/7/25

'''
sigmoid : 映射到概率的函数
model : 返回预测结果值
cost : 根据参数计算损失
gradient : 计算每个参数的梯度方向
descent : 进行参数更新
accuracy: 计算精度
'''

import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def sigmoid(z):
    return 1/(1+np.exp(-z))

def model(X,theta):
    return sigmoid(np.dot(X,theta.T))

def cost(X,y,theta):
    left = np.multiply(-y,np.log(model(X,theta)))
    right = np.multiply((1-y),np.log(1-model(X,theta)))
    return np.sum(left-right)/len(X)

def gradient(X,y,theta):
    grad = np.zeros(theta.shape)
    error = (model(X,theta)-y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error,X[:,j])
        grad[0,j] = np.sum(term)/len(X)
    return grad

def stopGriterion(stop_type,value,threshold):
    if stop_type == STOP_ITER :
        return value > threshold
    elif stop_type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif stop_type == STOP_GRAD:
        return np.linalg.norm(value) < threshold

def shuffleData(data):
    np.random.shuffle(data)
    col = data.shape[-1]
    X = data[:,:col-1]
    y = data[:,col-1:]
    return X,y

def descent(data,theta,batchSize,stopType,thresh,alpha):

    init_time = time.time()
    count = 0
    k = 0
    X,y = shuffleData(data)
    grad = np.zeros(theta.shape)
    costs = [cost(X,y,theta)]
    n = len(data)
    while True:
        grad = gradient(X[k:k+batchSize],y[k:k+batchSize],theta)
        k = k+batchSize

        if k >= n:
            k = 0
            X,y = shuffleData(data)
        theta = theta - alpha*grad
        costs.append(cost(X,y,theta))
        count = count+1

        if stopType == STOP_ITER:
            value = count
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopGriterion(stopType,value,thresh): break

    return theta,count-1,costs,grad,time.time()-init_time

def runExpe(data,theta,batchSize,stopType,thresh,alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    n = len(data)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize == n:
        strDescType = "Gradient"
    elif batchSize == 1:
        strDescType = "Stochastic"
    else:
        strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER:
        strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST:
        strStop = "costs change < {}".format(thresh)
    else:
        strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.show()
    return theta

def predict(X_test,theta):
    return [1 if x >= 0.5 else 0 for x in model(X_test, theta)]

if __name__ == '__main__':
    dataset = pd.read_csv("logic.csv",header=None,names=['Exam1','Exam2','Admitted'])
    dataset.insert(0,'Ones',1)
    data = dataset.as_matrix()
    col = data.shape[1]
    X = data[:,:col-1]
    y = data[:,col-1]
    n = len(data)
    theta = np.zeros((1,col-1))
    theta = runExpe(data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)


