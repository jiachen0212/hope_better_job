import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

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

ax.cla()
ax.set_title("LS")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.xlim((0, 20))
plt.ylim((0, 50))
plt.grid(True)
#points
ax.scatter(data_X, data_Y, color='k')
#LS fit line
ax.plot(data_X, YLS_pred, color='b', linewidth=3)

plt.legend([
    'LS fit line',
    'point data',
])
plt.pause(0.001)
plt.show()