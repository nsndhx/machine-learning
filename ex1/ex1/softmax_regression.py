# coding=utf-8
import numpy as np

def _loss(_x,_theta):
    z=np.dot(_x,_theta)
    z=np.exp(z)
    #计算每个样本对应0~9的概率之和
    sum=np.sum(z,axis=1,keepdims=True)
    #将样本对应0~9的概率除以0~9的概率和，使得最终概率和为1
    return z/sum#y[10,60000],probs[60000,10],x[60000,784]

def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta

    #theta: k*n矩阵  x：m*n矩阵   y：k*m矩阵
    #g(x)=x[k,m]*theta[m,n]=np.matmul(x,theta.T)
    print(y.shape[0],y.shape[1])
    for i in range(iters):
        print("第",i,"轮：")
        #输入数据的特征矩阵乘以参数矩阵，得每个样本特征对应0~9的概率
        probs=_loss(x,theta.T)
        #计算对应的损失值
        loss=-(1.0/x.shape[0])*np.sum(y.T*np.log(probs))
        print("loss:",loss)
        #计算该次迭代的梯度，并加入正则化项
        dw=-(1.0/x.shape[0])*np.dot(y-probs.T,x)+0.05*theta
        #print("dw:",dw)
        #更新theta参数
        theta-=alpha*dw
    
    return theta
    
