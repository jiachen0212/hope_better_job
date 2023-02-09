# coding=utf-8
# k个数组的最小公共区间
# 构建最小堆做
# 看个数组，维护一个k小堆，一直删除堆顶然后加入新的元素知道某个数组删完了，
# 最小堆的范围也就是公共区间了
# https://www.cnblogs.com/kira2will/p/4019588.html

a = [4,10,15,24,26]
b = [0,9,12,20]
c = [5,18,22,30]

la = len(a)
lb = len(b)
lc = len(c)
i,j,k =0,0,0
dui = [a[i], b[j], c[k]]
while (i<la and j<lb and k<lc):
    dui.sort()
    num = dui.pop(0)
    if num in a:
        if i+1<la:
            dui.append(a[i+1])
            i += 1
        else:
            print(num, dui[-1])
            break
    elif num in b:
        if j+1<lb:
            dui.append(b[j+1])
            j += 1
        else:
            print(num, dui[-1])
            break
    else:
        if k+1<lc:
            dui.append(c[k+1])
            k += 1
        else:
            print(num, dui[-1])
            break




