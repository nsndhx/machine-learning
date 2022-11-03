# coding=utf-8
import numpy as np

def predict(test_images, theta):
    scores = np.dot(test_images, theta.T)
    preds = np.argmax(scores, axis=1)
    return preds

def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    #输出分类器在测试集上的准确率
    a=y_pred.size
    amount=np.zeros(10)
    error=np.zeros(10)
    for i in range(a):
        if y_pred[i]!=y[i]:
            error[y[i]]+=1
        amount[y[i]]+=1
    for i in range(9):
        print((amount[i]-error[i])/amount[i])
    err=np.sum(error)
    return (a-err)/a
