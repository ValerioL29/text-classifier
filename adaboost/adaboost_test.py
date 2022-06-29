from numpy import *
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import pickle

with open('../preprocess/train/train_bag.pickle', 'rb') as file_obj:
    train_bunch = pickle.load(file_obj)
with open('../preprocess/test/test_bag.pickle', 'rb') as file_obj:
    test_bunch = pickle.load(file_obj)

y_train = train_bunch.label
X_train = train_bunch.tfidf_weight_matrices

# 分类器定义
Classifiers = [DecisionTreeClassifier(), MultinomialNB(alpha=0.001), LinearSVC(C=1, tol=1e-5)]

def buildStump(dataArr, classLabels, D,):
    labelMat = array(classLabels).T
    m, n = shape(dataArr)
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf  # 无穷
    for classifier in Classifiers:
        classifier.fit(dataArr, classLabels)
        predictedVals = classifier.predict(dataArr)
        errArr = mat(ones((m, 1)))
        errArr[predictedVals == labelMat] = 0  # 预测值与实际值相同，误差置为0
        weightedEroor = D.T * errArr  # D就是每个样本点的权值，随着迭代，它会变化，这段代码是误差率的公式
        if weightedEroor < minError:  # 选出分类误差最小的基分类器
            minError = weightedEroor  # 保存分类器的分类误差
            bestClassEst = predictedVals.copy()  # 保存分类器分类的结果
            bestStump['Classifier'] = classifier
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataMat, classLabels, numIt=40):  # 迭代40次，直至误差满足要求，或达到40次迭代
    weakClassArr = []  # 保存每个基分类器的信息，存入列表
    m = shape(dataMat)[0]
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        print("Starting Round")
        bestStump, error, classEst = buildStump(dataMat, classLabels, D)
        alpha = float(0.5 * log((1.0 - error) / error))  # 对应公式 a = 0.5* (1-e)/e
        print("Alpha weight is " + str(alpha))
        weakClassArr.append(bestStump['Classifier'])  # 把每个基分类器存入列表
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # multiply是对应元素相乘
        D = multiply(D, exp(expon))  # 根据公式 w^m+1 = w^m (e^-a*y^i*G)/Z^m
        D = D / D.sum()  # 归一化
        aggClassEst += alpha * classEst  # 分类函数 f(x) = a1 * G1
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # 分错的矩阵
        errorRate = aggErrors.sum() / m  # 分错的个数除以总数，就是分类误差率
        print('total error: ', errorRate)
        if errorRate == 0.0:  # 误差率满足要求，则break退出
            break
    return weakClassArr, aggClassEst


if __name__ == '__main__':  # 运行函数

    weakClassArr, aggClassEst = adaBoostTrainDS(X_train, y_train)

    # print('weakClassArr', weakClassArr)
    # print('aggClassEst', aggClassEst)
    # classify_result = adaClassify([0.7, 1.7], weakClassArr)  # 预测的分类结果，测试集我们用的是[0.7,1.7]测试集随便选
    # print("结果是:", classify_result)