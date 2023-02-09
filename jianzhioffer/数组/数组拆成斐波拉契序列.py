# coding=utf-8
# 数组拆分成斐波拉契序列
'''
输入: "11235813"
: [1,1,2,3,5,8,13]
输出: True

输入："123456579"
输出：[123,456,579]
True
'''

# 回溯法
class Solution:
    def splitIntoFibonacci(self, S):
        if not S:
            return []
        res = []  # 构成的斐波拉契序列
        l = len(S)
        def helper(S, res, idx):
            if idx == l and len(res) >= 3:
                return True
            for i in range(idx, l):
                if S[idx] == '0' and i > idx:  # '03'这种  直接False
                    break
                num = int(S[idx: i+1])
                tmp_n = len(res)
                if num > 2**31-1:  # 不能超过最大整数
                    break
                if tmp_n >= 2 and num > res[-1] + res[-2]: # 前两个数之和大于现在的数
                    break
                if tmp_n <= 1 or num == res[-1] + res[-2]:
                    res.append(num)
                    if helper(S, res, i+1):
                        return True
                    res.pop()
            return False
        helper(S, res, 0)
        return res


'''
几乎一样的题：
输入: "199100199"
输出: true
解释: 累加序列为: 1, 99, 100, 199。1 + 99 = 100, 99 + 100 = 199
'''
class Solution:
    def isAdditiveNumber(self, nums):
        if not nums:
            return False
        res = []  # 构成的斐波拉契序列
        l = len(nums)
        def helper(idx):
            if idx == l and len(res) >= 3:
                return True
            for i in range(idx, l):
                if nums[idx] == '0' and i > idx:  # '03'这种  直接False
                    break
                num = int(nums[idx: i+1])
                tmp_n = len(res)
                # if num > 2**31-1:  # 不能超过最大整数
                #     break
                if tmp_n >= 2 and num > res[-1] + res[-2]: # 前两个数之和大于现在的数
                    break
                if tmp_n <= 1 or num == res[-1] + res[-2]:
                    res.append(num)
                    if helper(i+1):
                        return True
                    res.pop()
            return False
        return helper(0)