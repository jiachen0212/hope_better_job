# coding=utf-8
# 数组中消失的数
# leetcode ac
'''
输入:
[4,3,2,7,8,2,3,1]

输出:
[5,6]

要求：时复：O(n) 空复O(1)
'''

'''
思路：
将所有数组元素作为数组下标，置对应数组值为负值。
那么，仍为正数的位置即为（未出现过）消失的数字。
(好好理解下！！)
'''
class Solution(object):
    def findDisappearedNumbers(self, nums):
        res = []
        l = len(nums)
        for i in range(l):
            # 注意这里用的abs  因为不申请额外的空间，所以在置复数期间会
            # 有些元素提前变负数了，所以需要用下abs!!!
            nums[abs(nums[i])-1] = -(abs(nums[abs(nums[i])-1]))
        for j in range(l):
            if nums[j] > 0:
                res.append(j+1)
        return res
s = Solution()
print(s.findDisappearedNumbers([4,3,2,7,8,2,3,1]))