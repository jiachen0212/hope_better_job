# coding=utf-8
#### 数组中第k大的元素
# 最小堆做

def dui(s, k):
    kk = []
    for i in range(k):
        kk.append(s[i])

    # 另起一个循环,避免重复添加某一元素..
    for i in range(k, len(s)):
        if s[i] > min(kk):  # 把kk中的最小数不断替换掉,加入新的大数
            # kk.pop(kk.index(min(kk)))
            kk.remove(min(kk))
            kk.append(s[i])
    return min(kk)


a = [5,0,7,2,1,7,8,7,9]
print dui(a, 4)

