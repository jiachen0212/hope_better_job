#coding=utf-8
# 寻找第n个丑数  丑数:即因子里只有2 3 5 的数. 1是第一个丑数

# [2, 3, 5...] 遍历现有的数组,分别寻找到,乘以2 3 5 后,大于当前最大丑数的三个数: M2 M3 M5. 取M2 M3 M5 中最小的那个作为新找到的那个丑数
# 即现有的丑数数组是[1, 2, 3, 5 ...M] 则计算后就变成[1, 2, 3, 5 ...M, min(M2, M3, M5)]


# 牛客 ac
# -*- coding:utf-8 -*-
# 第index个丑数
class Solution:
    def GetUglyNumber(self, index):
        record = [1]  # 存放所有的丑数
        if index == 1:
            return record[0]
        if index == 0:
            return 0
        index2, index3, index5 = 0, 0, 0
        # 因为丑数只含2 3 5 这3个因子
        M2, M3, M5 = record[index2] * 2, record[index3] * 3, record[index5] * 5
        # while len(record) == n:
        while len(record) < index:
            while M2 <= record[-1]:  # record[-1]代表当前最大的那么丑数M
                index2 += 1
                M2 = record[index2] * 2 # 跳出此while循环的时候,表明找到了第一个乘以2后大于当前最大丑数的M2
            while M3 <= record[-1]:
                index3 += 1
                M3 = record[index3] * 3
            while M5 <= record[-1]:
                index5 += 1
                M5 = record[index5] * 5
            record.append(min(M2, M3, M5))  # min(M2, M3, M5) 把M2,M3,M5中最小的那个作为新找到的那个新丑数,放入record
        return record[-1]

s = Solution()
res = s.GetUglyNumber(10)
print res



# 超级丑数
# leetcode  ac
def GetUglyNumber(index, primes):
    record = [1]  # 存放所有的丑数
    if index == 1:
        return record[0]
    if index == 0:
        return 0

    l = len(primes)
    Ms = []
    indexs = [0]*l
    for i in range(l):
        Ms.append(record[0] * primes[i])

    nums = []
    while len(record) < index:
        for i in range(l):
            while Ms[i] <= record[-1]:
                indexs[i] += 1
                Ms[i] = record[indexs[i]] * primes[i]
        record.append(min(Ms))  # min(M2, M3, M5) 把M2,M3,M5中最小的那个作为新找到的那个新丑数,放入record
    return record[-1]
ans = GetUglyNumber(12, [2,7,13,19])
print(ans)