# coding=utf-8
# n个有序数组topk
# 堆排序
# 大顶堆
# https://www.cnblogs.com/ywl925/p/3794852.html
# 类似n个数组的公共区间问题 用的是最小堆做的

# 删除最大堆堆顶，保存到数组或者栈中，然后向最大堆插入删除的元素所在数组的下一个元素

# example 3 lists find top6:
a = [26, 24, 15, 10, 4]
b = [20, 12, 9, 6, 8]
c = [23, 22, 19, 5, 2]
d = [18, 16, 13, 11, 3]
k = 16

def fun(a,b,c,d,k):
    dadingdui = [a[0], b[0], c[0], d[0]]
    dadingdui.sort()
    res = []
    res.append(dadingdui.pop())  # res添加第一个top值
    while len(res) < k:
        if res[-1] in a:
            if a[a.index(res[-1])+1:]:
                # 删除最大堆堆顶，保存到数组或者栈中，然后向最大堆插入删除的元素所在数组的下一个元素
                dadingdui.append(a[a.index(res[-1])+1])
        if res[-1] in b:
            if b[b.index(res[-1])+1:]:
                dadingdui.append(b[b.index(res[-1])+1])
        if res[-1] in c:
            if c[c.index(res[-1])+1:]:
                dadingdui.append(c[c.index(res[-1])+1])
        if res[-1] in d:
            if d[d.index(res[-1])+1:]:
                dadingdui.append(d[d.index(res[-1])+1])
        dadingdui.sort()
        res.append(dadingdui.pop())
    return res

res = fun(a,b,c,d,k)
print res






######  堆实现代码
# 测试的话就让k是可以整除n的 这样初始堆好分配些

# 维护一个大顶堆  也就是父>子 目的是堆的首元素最大
def heap_adjust(data, root):
    if 2*root+1 < len(data):
        # 下面四行 找到更大的子
        if 2*root+2 < len(data) and data[2*root+2] > data[2*root+1]:
            k = 2*root+2
        else:
            k = 2*root+1
        # 把堆顶换成最大
        if data[k] > data[root]:
            data[k],data[root] = data[root],data[k]
            heap_adjust(data,k)  # 递归维护下一个层子

def min_heap(data):
    ind = len(data)/2 - 1
    for i in range(ind, -1, -1):
        heap_adjust(data, i)
    return data

def N_top_k(a,b,c,d,k):
    val = k / 4
    data_k = a[:val] + b[:val] + c[:val] + d[:val]
    data_k = min_heap(data_k)  # data_k[-1]最大
    top_k = []
    tmp_a = a[val:]
    tmp_b = b[val:]
    tmp_c = c[val:]
    tmp_d = d[val:]
    while k:
        tmp_top = data_k.pop(0)  # 大顶堆  所以最大在index=0处
        top_k.append(tmp_top)
        k -= 1
        if tmp_top in a:
            if tmp_a and a.index(tmp_top): # 删除最大堆堆顶，保存到数组或者栈中，然后向最大堆插入删除的元素所在数组的下一个元素
                data_k.append(tmp_a.pop(0))
        if tmp_top in b:
            if tmp_b and b.index(tmp_top):
                data_k.append(tmp_b.pop(0))
        if tmp_top in c:
            if tmp_c and c.index(tmp_top):
                data_k.append(tmp_c.pop(0))
        if tmp_top in d:
            if tmp_d and d.index(tmp_top):
                data_k.append(tmp_d.pop(0))
        data_k = min_heap(data_k) # 维护大顶堆
    return top_k

a = [26, 24, 15, 10, 4]
b = [20, 12, 9, 6, 8]
c = [23, 22, 19, 5, 2]
d = [18, 16, 13, 11, 3]
k = 8
print N_top_k(a,b,c,d,k)





