# coding=utf-8
# leetcode 版本的
# 和判断数组中消失的数一样，也是用取负来做
# 取负然后放入输出矩阵，重复的数就会出现第二次要放的时候位置上已经有数了，那么就找到了它
# 1 ≤ a[i] ≤ n  1~n的元素值

class Solution(object):
    def findDuplicates(self, numbers):
        res = []
        if numbers == None:
            return res
        l = len(numbers)
        out = [0]*l  # 没访问的到，标记为0
        for i in range(l):
            if out[numbers[i]-1] == 0:
                out[numbers[i]-1] = -numbers[i]  # 这里给别的值其实也可以
                # 为了延续数组消失数的思想，还是用-nums[i]吧～ 表示index numbers[i]-1处已经被访问好了
            else:
                res.append(numbers[i])
        return res

s = Solution()
print(s.findDuplicates([4,3,2,7,8,2,3,1]), '+++')

# 改版：
'''
如果数大小是0～n，数组长是n，那么注意把输出数组长度加大一位就ok了，其他思想一样
class Solution(object):
    def findDuplicates(self, numbers):
        res = []
        if numbers == None:
            return res
        l = len(numbers)
        out = [0]*(l+1)
        for i in range(l):
            if out[numbers[i]] == 0:
                out[numbers[i]] = -numbers[i]  # 这里给别的值其实也可以
                # 为了延续数组消失数的思想，还是用-nums[i]吧～
            else:
                res.append(numbers[i])
        return res

s = Solution()
print(s.findDuplicates([0,3,1,2,2,6]), '===')
'''



# -*- coding:utf-8 -*-
####### 牛客版本的
####### 题目不是很一样....

class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        if numbers == None:
            return False
        l = len(numbers)
        for i in range(l):
            while numbers[i] != i:
                if numbers[numbers[i]] == numbers[i]:
                    duplication[0] = numbers[i]
                    return True
                else:
                    cache = numbers[i]
                    numbers[i] = numbers[cache]
                    numbers[cache] = cache  # 这三行保证了num[i]index处
                    # num[index]==index
        return False


s = Solution()
res = s.duplicate([5,3,1,0,2,5,3], [0,0,0])
print(res)