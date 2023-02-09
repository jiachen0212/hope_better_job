#coding=utf-8
# 寻找数组中所有的逆序对数..
# 每次寻找当前list中的最大值,然后统计其后的数量.再把这个max值删掉,接着找次max和统计其后的数量.重复以上步骤...

# 方法一:
def fun(s):
    count = 0
    l = len(s)
    while s != []:
        mmax = max(s)
        ind = s.index(mmax)
        count += l - 1 - ind
        s.pop(ind) # 把当前的max值减掉
        l -= 1
    return count

res = fun([7,5,6,4])
# print res





########### 提交牛客  有点超时   部分ac

def InversePairs(data):
    count = 0
    copy = []
    for i in data:
        copy.append(i)
    copy.sort()  # 升序， 即从小到大
    for i in range(len(copy)):
        count += data.index(copy[i])  # 注意这里是在data里找index
        # 即看看当前data中最大的数后面有小的数，位置在哪。 会被多少个前面的数超过它，即可形成多少对逆序对
        data.remove(copy[i])
    return count

res = InversePairs([7,5,6,4])
