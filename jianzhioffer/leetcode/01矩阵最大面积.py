# coding=utf-8

# 找出0-1矩阵里，全1组成的最大矩形面积
# https://blog.csdn.net/hopeztm/article/details/7870387
# 思路：逐行扫每一行，统计其下方为1的元素个数，类似一个histogram高度
# 这样得到了新的矩阵了，就看每一行连续和最大就ok了
# leetcode diff  https://leetcode-cn.com/problems/maximal-rectangle/submissions/

def maximalRectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    n = len(matrix[0])
    height = [0] * (n + 1)
    ans = 0
    for row in matrix:
        for j in range(n):   # [i][j]==1则height+1,否则height清0
            height[j] = height[j] + 1 if row[j] == 1 else 0
        stack = [-1]   # stack存放的是width信息
        # stack会依次压入列的index，当height[stack[-1]]<height[r]
        # 证明后一列下方的高度减小了，这个时候把之前那个较大的height作为h
        # 宽度就是当前列index r-1 - stack[-1] 很好理解啊这个算法
        for r in range(n + 1):
            while height[r] < height[stack[-1]]:
                h = height[stack.pop()]
                w = r - 1 - stack[-1]
                ans = max(ans, h * w)
            stack.append(r)  # 不管上面的while循环,每次反正会把r压进来
            # 缺失的r就是上面被pop掉了
    return ans

matrix = [[0,1,0,1,0,0],[0,1,1,0,0,1],[1,1,1,0,1,0],[0,1,1,0,0,1]]
res = maximalRectangle(matrix)
print(res)
