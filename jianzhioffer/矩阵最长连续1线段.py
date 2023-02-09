# coding=utf-8
# 矩阵中最长的连续1线段
# https://www.cnblogs.com/grandyang/p/6900866.html

# 类似DFS思想，当遇到了1，就查看他的水平竖直对角4个邻，看是0/1 并计数1的个数

def fun(M):
    if not M or not M[0]:
        return 0
    m = len(M)
    n = len(M[0])
    dicts = [[1,0],[0,1],[-1,-1],[-1,1]]
    max_count = 1
    for i in range(m):
        for j in range(n):
            if M[i][j] == 0:
                continue
            for k in range(4):   # 水平 竖直 对角 逆对角
                count = 0  # 注意count是在k循环内，所以每个count是针对每一个方向计数的
                x,y = i,j
                while (x>=0 and y>=0 and x<m and y<n and M[x][y] == 1):
                    x += dicts[k][0]
                    y += dicts[k][1]  # 在四个邻上search
                    count += 1
                max_count = max(max_count, count)
    return max_count


a = [[1,1,1,0,1],
 [1,1,1,0,1],
 [0,1,1,0,1],
 [0,0,0,1,0]]

print(fun(a))

