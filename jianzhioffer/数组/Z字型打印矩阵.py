# coding=utf-8
# Z 字型打印矩阵

'''
   每层的索引和:
            0:              (00)
            1:            (10)(01)
            2:          (02)(11)(20)
            3:          (30)(21)(12)
            4:          (22)(31)(40)
发现规律了吧，xyindex之和是层index
'''

s = [[1,3,4,10,11],[2,5,9,12,19],[6,8,13,18,20],[7,14,17,21,24],[15,16,22,23,25]]
# s = [[1,3,4,10],[2,5,9,11],[6,8,12,15],[7,13,14,16]]
# s = [[1,3,4],[2,5,8],[6,7,9]]
# s = [[1,3],[2,4]]
# s = [[1,3,4,10,11,21],[2,5,9,12,20,22],[6,8,13,19,23,30],[7,14,18,24,29,31],[15,17,25,28,32,35],[16,26,27,33,34,36]]
res = []
flag = 0

# 借鉴leetcode的对角打印矩阵思想
def fun(matrix):
    n = len(matrix) - 1     # 横轴 索引最大值 n
    m = len(matrix[0]) - 1  # 纵轴 索引最大值 m
    c = n + m + 1           # 层数 等于 横纵最大索引之和 + 1
    l = []
    for x in range(c+1):    # 每层的横纵索引之和相等，刚好等于 层数值+1
        if x % 2 == 0:      # 索引和为{偶}数，向上遍历，{横}索引值递减，遍历值依次是(x,0),(x-1,1),(x-2,2),...,(0,x)，不要索引出界的，即可
            for i in range(x+1):
                j = x - i
                if i <= n and j <= m:
                    l.append(matrix[i][j])
                elif i > n:
                    break
                else:
                    continue
        else:
            for j in range(x+1):
                i = x - j
                if i <= n and j <= m:
                    l.append(matrix[i][j])
                elif j > m:
                    break
                else:
                    continue
    return l





# 这是硬撸出来的 没技巧
def fun1(s):
    a,b = 0,0
    ll = len(s)   # 默认觉得他是方阵好了
    if ll==1:
        return s[-1][-1]
    res.append(s[a][b])
    while a < 2 * (ll / 2):
        if a+1 < len(s):
            res.append(s[a+1][b])
            for i in range(1, a+1 + 1):
                res.append(s[a+1 - i][b+i])
            if b + a+1 +1 < len(s[0]):
                flag = 1
                res.append(s[0][b+a+1 +1])
                for j in range(1, b+a+1+1 + 1):
                    res.append(s[0+j][b+a+1+1 - j])
            else:
                flag = 0
        a += 2
    if flag:    # 此时的打印到了左下角
        while b < 2 *(ll / 2):
            if b+1 < len(s[0]):
                res.append(s[a][b+1])
                for i in range(1, ll-1-b):
                    res.append(s[a-i][b+1+i])
                if b+1 +1 < ll:
                    res.append(s[b+1 + 1][ll-1])
                    for j in range(1, ll-b-2):
                        res.append(s[b+2+j][ll-1-j])
            b += 2
    else:   # 此时打印到了右上角
        a, b = 0, ll-1
        if ll==2:
            res.append(s[-1][-1])
        while a < 2 * (ll / 2 - 1):
            res.append(s[a+1][b])
            for i in range(1, ll-a-1):
                res.append(s[a+1+i][ll-1-i])
            if a+1 + 1 < ll:
                res.append(s[ll-1][a+1+1])
                for j in range(1, ll-a-1-1):
                    res.append(s[ll-1-j][a+1+1 +j])
                if a + 3 < ll:
                    res.append(s[a+3][ll-1])
            a += 2
    return res


# ans1 = fun1(s)
ans = fun(s)
print(ans)
# print(ans1)


