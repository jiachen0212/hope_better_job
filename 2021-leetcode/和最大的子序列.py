#coding=utf-8
# 获得和最大的连续子序列  即得是连续的元素,组成的子序列,子序列的sum最大. 栗子:[1, -2, 3, 10, -4, 7, 2, -5]的最大和子序列是: [3, 10, -4, 7, 2]
# 方法一: 当当前的sum<=0时候,则摒弃这前一段的子序列,只继续看后面的序列. 当sum>0则继续遍历后面的元素. 存储并且不断更新当前得到的最大值.
# 方法二: 动态规划,递归实现.
# f(i-1)表示前i-1个数组合起来的最大和.当f(i-1)<=0,则f(i)=s[i]本身;递归遍历每一个i的f(i),并设置变量maxsum存储最大的sum值,即可得到最大sum..
# 因为fun计算的是到当前位置的max值,所以当当前位置是负数时会把summax值减小. 所以使用一个变量存储最大sum并更新它...
import random


# 获得随机数组
def getrandmodata(num):
    s = []
    for i in range(num):
        s.append(random.randint(-10, 10))
    return s


########################### 方法一 ############################
def getmaxsumsubsque(s):
    if len(s) <= 0:
        print 'empty list...'
        return False, False  # 因为我的函数输出要求的是res1, subs两个,所以我这里得写两个False.
    p1 = p2 = 0  # 指向最大和子序列的首末元素
    P1 = P2 = 0  # 指向最终和最大的子序列的,首末元素
    maxsum = s[0]  # 初始化最大和值
    j = 0  # j用来计数遍历完整个list. 不能使用p2的指向作为list是否遍历完的判断. 因为p2是指向最大和子序列的末尾,它不一定值会等于list的尾.
    while j < len(s) and p2 < len(s):
        j += 1
        cursum = sum(s[p1:p2+1])
        if cursum > maxsum:
            P1 = p1
            P2 = p2
            maxsum = cursum  # 更新已经计算过的最大的子序列和
        if cursum > 0:  # 这里设置成>0,也就是当前面的子序列和=0时也把前面的给丢掉.
            p2 += 1
        else:  # 当前得到的和是负数,那么就摒弃之前的所有子序列.
            p1 = p2 + 1
            p2 = p1
    return maxsum, s[P1:P2+1]


########################### 方法二 ############################
# 动态规划... 递归实现
# f(i-1)表示前i-1个数组合起来的最大和.当f(i-1)<=0,则f(i)=s[i]本身;递归遍历每一个i的f(i),并设置变量maxsum存储最大的sum值,即可得到最大sum..
# 因为fun计算的是到当前位置的max值,所以当当前位置是负数时会把summax值减小. 所以使用一个变量存储最大sum并更新它...
class Solution:
    def fun(self, i, fre, a, maxsum):
        if fre <= 0 or i == 0:
            # print 'fre<0, fre=:', fre, 'a = :', a
            fcur = a
        if i > 0 and fre > 0:
            # print 'fre>0, fre=:', fre, 'a = :', a
            fcur = fre + a
        if fcur > maxsum:  # 因为fun计算的是计算到当前位置的max,所以当当前位置是负数时会把max值减小.所以使用一个变量存储maxsum...
            maxsum = fcur
        # print '当前最大子序列和:', fcur
        return fcur, maxsum

    def maxsumofSubarray(self, s):
        if len(s) <= 0:
            return False
        fre = s[0]
        maxsum = s[0]
        for i in range(len(s)):
            fre, maxsum = self.fun(i, fre, s[i], maxsum)
        return maxsum

s = getrandmodata(10)

res1, subs = getmaxsumsubsque(s)
res2 = maxsumofSubarray(s)

print '方法一:', res1, subs
print '方法二:', res2



