import numpy as np
import random
import math
import matplotlib.pyplot as plt
'''
RANSAC: 标记内外点, 内点参与拟合直线, 外点视为异常点 
内点集拟合出直线(y=k*x+b), 然后计算剩余点到直线的dis, 过了阈值则是外点, 阈值内则视为内点加入拟合直线算kb
'''
 
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
 
#region data
N = 40
X = np.linspace(start = 0, stop = 20, num = N)
k = b = 2
Y = k * X + b + np.random.randn(N) * 3
 
data_X = []
data_Y = []
for i in range(N):
    data_X.append(X[i])
    data_Y.append(Y[i])
    
#添加粗差
for i in range(5):
    data_X.append(i+15)
    data_Y.append(1)
 
N = len(data_X)
#endregion
 
#region 直接用最小二乘拟合直线
# https://blog.csdn.net/xinjiang666/article/details/103782544
kLS = bLS = 0
sumX = sumY = 0
sumXX = sumXY = 0
for i in range(N):
    xi = round(data_X[i])
    yi = round(data_Y[i])
    sumX += xi
    sumY += yi
    sumXX += xi * xi
    sumXY += xi * yi
deltaX = sumXX * N - sumX * sumX
 
if (abs(deltaX) > 1e-15):
    kLS = (sumXY * N - sumX * sumY) / deltaX
    bLS = (sumXX * sumY - sumX * sumXY) / deltaX
 
YLS_pred = kLS * np.asarray(data_X) + bLS
#endregion
 
#region RANSAC
#region 设置先验参数或期望
n = M = 2           #模型拟合所需的最小点数
Iter_Number = 1000  #最大迭代次数
DisT = 3            #点到直线的距离阈值
ProT = 0.95         #期望的最小内点比例
#endregion
 
best_k = 0
best_b = 0
best_inliers_number = N*0.1
best_X_inliers=[]
best_Y_inliers=[]
iter = 0
existed_k_b=[] #已经使用过的k、b
 
#region 5 迭代
for i in range(Iter_Number):
    iter += 1
 
    #region 1 随机取M个点用于模型估计
    indx1, indx2 = random.sample(range(len(data_X)), M)
    point1_x = data_X[indx1]
    point1_y = data_Y[indx1]
    point2_x = data_X[indx2]
    point2_y = data_Y[indx2]
    #endregion
 
    #region 2 计算模型参数
    k = round((point2_y - point1_y) / (point2_x - point1_x), 3)
    b = round(point2_y - k * point2_x, 3)
    current_k_b = [k,b]
    if current_k_b in existed_k_b:#防止重复的两个点或参数用于计算
        continue
    else:
        existed_k_b.append([k,b])
    #endregion
 
    #region 3 计算内点数量（或比例）
    X_inliers = []
    Y_inliers = []
    inliers_current = 0 #当前模型的内点数量
    for j in range(len(data_X)):
        x_current = data_X[j]
        y_current = data_Y[j]
        dis_current = abs(k * x_current - y_current + b) / math.sqrt(k * k + 1) # 点到拟合直线的距离dis_current
        if (dis_current <= DisT):
            inliers_current += 1
            X_inliers.append(x_current)
            Y_inliers.append(y_current)
    print("第{}次, 内点数量={}, 最佳内点数量={}"
          .format(iter, inliers_current, best_inliers_number))
    #endregion
 
    #region 4 如果当前模型的内点数量大于之前最好的模型的内点数量，跟新模型参数和迭代次数
    if (N>inliers_current >= best_inliers_number):
        Pro_current = inliers_current / N       #当前模型的内点比例Pro_current
        best_inliers_number = inliers_current   #更新最优内点的数量
        best_k = k  #更新模型参数
        best_b = b  #更新模型参数
        i = 0       #当前迭代置为0
        Iter_Number = math.log(1 - ProT) / math.log(1 - pow(Pro_current, M))    #更新迭代次数
        best_X_inliers = X_inliers  #更新内点
        best_Y_inliers = Y_inliers  #更新内点
        print("更新结果：k={}, b={}, 当前内点比例={}, 最佳内点比例={}, 新的迭代次数={}"
              .format(best_k, best_b, Pro_current, best_inliers_number/N, Iter_Number))
    # endregion
 
    #region plot
    ax.cla()
    ax.set_title("RANSAC and LS")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.xlim((0, 20))
    plt.ylim((0, 50))
    plt.grid(True)
 
    #points
    ax.scatter(data_X, data_Y, color='k')
 
    #LS fit line
    ax.plot(data_X, YLS_pred, color='b', linewidth=3)
 
    #RANSAC fit line
    Y_R_pred = best_k * np.asarray(data_X) + best_b
    ax.plot(data_X, Y_R_pred, color='g', linewidth=5)
 
    plt.legend([
        'LS fit line',
        'RANSAC fit line',
        'point data',
    ])
    plt.pause(0.001)
    #endregion
 
    #region XXX 如果最佳的内点比例大于期望比例，则跳出
    if ((best_inliers_number / N) > ProT):
        print("终止：内点比例=", (best_inliers_number / N), "大于 期望内点比例=", ProT)
        break
    #endregion
 
#endregion
#endregion
 
#region plot inliers
plt.plot(best_X_inliers, best_Y_inliers, "ro")
plt.legend([
    'LS',
    'RANSAC',
    'inlier',
    'outlier',
])
plt.show()
#endregion
