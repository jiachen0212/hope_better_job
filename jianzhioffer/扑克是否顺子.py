#coding=utf-8
# 判断5张牌是不是顺子(即连续) 其中大小王可以代表任意数 把大小王抽象为0值 其他则是: 1~13 A-K
# 思路: 首先对5张牌排序,并要保证数组内无重复元素(除0外);然后统计数组内0的个数;最后计算数组中相邻元素的差值,差值之和大于0的个数的话则不满足顺子要求.


# 把扑克牌值转变成数值
def change(s):
    for i in range(len(s)):
        if s[i] == 'A':
            s[i] = 1
        if s[i] == 'J':
            s[i] = 11
        if s[i] == 'Q':
            s[i] = 12
        if s[i] == 'K':
            s[i] = 13
    return s

def wheath_shunzi(s):
    if len(s) < 5:
        return False
    s = change(s)
    s.sort()  # 对s进行顺序排序
    # print s
    zerocon = 0
    gap = 0  # 数组内,所有相邻元素的间隔和
    for i in range(len(s)):
        if s[i] == 0:
            zerocon += 1
    small = zerocon
    big = small + 1   # 排除牌中0的index，后移一位正式进行相邻元素差值统计
    while big < len(s):
        if s[small] == s[big]:
            return False  # 出现对子,不可能是顺子了
        gap += s[big] - s[small] - 1
        small += 1
        big += 1
    if gap > zerocon:
        return False
    else:
        return True

res = wheath_shunzi([0, 10, 'K', 9, 0])
print res



######### 牛客 ac 版  和上面一模一样的其实...
# -*- coding:utf-8 -*-
class Solution:
    def change(self, s):
        for i in range(len(s)):
            if s[i] == 'A':
                s[i] = 1
            if s[i] == 'J':
                s[i] = 11
            if s[i] == 'Q':
                s[i] = 12
            if s[i] == 'K':
                s[i] = 13
        return s
    def IsContinuous(self, numbers):
        if numbers == []:
            return False
        numbers = self.change(numbers)
        scort_nums = list.sort(numbers)
        Wang = numbers.count(0)
        gap = 0
        cur = Wang
        later = cur + 1
        while later < len(numbers):
            if numbers[later] == numbers[cur]:
                return False
            gap += numbers[later] - numbers[cur] - 1
            cur += 1
            later += 1
        if gap > Wang:
            return False
        else:
            return True

s = Solution()
res1 = s.IsContinuous([0, 10, 'Q', 8, 0])
print(res1)
