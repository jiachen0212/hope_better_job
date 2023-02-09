# coding=utf-8
def fun(lis):
    for ind, num in enumerate(lis):
        if ind != num:
            if lis[num] == num:
                # 要替换的那个值等于现在位置上的值，证明找到了重复值
                doub = num
            tmp = lis[num]
            lis[ind] = tmp
            lis[num] = num
    return doub


dou = fun([2, 3, 1, 0, 3, 5, 3])
print(dou)