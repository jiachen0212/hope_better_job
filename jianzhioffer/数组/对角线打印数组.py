# coding=utf-8
# 对角线打印数组
'''
   每层的索引和:
            0:              (00)
            1:            (01)(10)
            2:          (20)(11)(02)
            3:        (03)(12)(21)(30)
            4:      (40)(31)(22)(13)(04)
            5:        (14)(23)(32)(41)
                           ... ...


发现规律了吧，xyindex之和是层index
'''

def print_matrix(s):
    res = []
    m = len(s)-1
    n = len(s[0])-1
    c = n+n+1   # 一共会有的层数
    for x in range(c+1):
        if x % 2 == 0:  # 偶数层, 第一个索引是递减 第二个索引用x-i就ok
            for i in range(x, -1, -1):
                j = x - i
                if i<=n and j<=m:  # 注意边界
                    res.append(s[i][j])
                elif j>m:
                    break
                else:
                    continue
        else:  # 奇数层   注意ij的位置
            for j in range(x, -1, -1):
                i = x - j
                if i<=n and j <=m:
                    res.append(s[i][j])
                elif i>n:
                    break
                else:
                    continue
    return res

s = [
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
ans = print_matrix(s)
print(ans)