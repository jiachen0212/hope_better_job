#coding=utf-8

# 用 ROC曲线算AUC
# import numpy as np
# import matplotlib.pyplot as plt
# def plotROC(predStrengths, classLabels):
#     cur = (1.0, 1.0)
#     ySum = 0.0  # 计算AUC
#     numPosClas = sum(np.array(classLabels) == 1.0) # 正例个数
#     yStep = 1 / float(numPosClas)
#     xStep = 1 / float(len(classLabels) - numPosClas)
#     sortedIndicies = predStrengths.argsort() # 对预测结果排序
#     fig = plt.figure()
#     fig.clf()
#     ax = plt.subplot(1, 1, 1)
#     # for index in sortedIndicies.tolist()[0]:
#     for index in range(len(sortedIndicies)):
#         if classLabels[index] == 1.0:
#             delX = 0; delY = yStep
#         else:
#             delX = xStep; delY = 0
#             ySum += cur[1]
#         ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], 'b')
#         cur = (cur[0] - delX, cur[1] - delY)  # 更新cur
#     # ax.plot([0, 1], [0, 1], 'b--')  # 这个是绘制随机结果的ROC曲线,直接是过0点k=1的直线
#     plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
#     plt.title('ROC curve')
#     ax.axis([0, 1, 0, 1])
#     plt.show()
#     print "the Area Under the Curve is: ", ySum*xStep


# pred = np.array([1, 1, 1, 1, -1, -1, 1, 1, -1, -1])
# label = [1, -1, 1, 1, -1, -1, 1, -1, 1, 1]
# plotROC(pred, label)






# 版本2  简洁的公式版本
def AUC(label, pre):
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]

    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5

    return float(auc) / (len(pos)*len(neg))


if __name__ == '__main__':
    label = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    pre = [0.9, 0.7, 0.6, 0.55, 0.52, 0.4, 0.38, 0.35, 0.31, 0.1]
    print(AUC(label, pre))

