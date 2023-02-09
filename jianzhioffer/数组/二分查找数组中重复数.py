#coding=utf-8
# 二分查找数组中重复的数字
# 数组n+1长,数字范围为1～n,必有一重复元素

def count(a, start, end):
    c = 0
    for i, m in enumerate(a):
        if m >= start and m <= end:
            c += 1
    return c


def fun(a):
    start = 1
    end = len(a) - 1
    while end >= start:
        mid = (end + start) / 2
        c = count(a, start, mid)  # 统计a中值范围在[start, mid]这区间内的个数
        if end == start:
            if c > 1:
                return start
            else:
                break
        if c > (mid + 1 - start):  # 理论上无重复的话，c应该==mid+1-start的
            # 但是这一段数量多了
            end = mid     # 所以重复的在前一段
        else:  # 重复数在第二段
            start = mid + 1


a = [1, 3, 5, 4, 3, 5, 6, 7]
# a = [1, 2, 2]
res = fun(a)
print(res)


