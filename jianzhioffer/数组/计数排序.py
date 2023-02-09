# coding=utf-8
# https://www.itcodemonkey.com/article/11750.html

# 计数排序

def count_sort(s):
    Max = max(s)
    Min = min(s)
    B = [0 for i in range(Min, Max+1)]
    for i in s:
        for j in range(Min, Max+1):
            if i == j:
                B[j - Min] += 1  # 计数

    # ok 已经计数好了，现在开始把排序打印出来
    C = []
    for i in range(len(B)):
        if B[i] > 0:
            while B[i] > 0:
                # print(i + Min)
                C.append(i + Min)
                B[i] -= 1
    return C

res = count_sort([2,5,3,0,2,3,0,3])
print(res)