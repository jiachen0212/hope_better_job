# coding=utf-8
# 连续子序列乘积最大
# 可以存在负数的...
'''
示例 1:
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。

示例 2:
输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
'''

# 这个思路好强！！！
'''
可以看成求被0拆分的各个子数组的最大值。

当一个数组中没有0存在，则分为两种情况：

1.负数为偶数个，则整个数组的各个值相乘为最大值；

2.负数为奇数个，则从左边开始，乘到最后一个负数停止有一个“最大值”，
从右边也有一个“最大值”，比较，得出最大值。
前后算两遍就ok

'''
class Solution(object):
    def maxProduct(self, nums):
        l = len(nums)
        mmax = nums[0]
        a = 1
        for i in range(l):
            a *= nums[i]
            mmax = a if a > mmax else mmax
            if nums[i] == 0:
                a = 1   # 遇到0 重新乘

        # 负数为偶数个，则整个数组的各个值相乘为最大值
# 负数为奇数个，则从左边开始，乘到最后一个负数停止有一个“最大值”，(中间的负负可得正)
# 同理从右边开始乘到最后一个负数，也有一个“最大值”，比较这俩最大值即可
# 所以只要左右分别遍历一回就ok
        a = 1
        for j in range(l-1,-1,-1):
            a *= nums[j]
            mmax = mmax if mmax > a else a
            if nums[j] == 0:
                a = 1
        return mmax

s = Solution()
res = s.maxProduct([-1,2,3,4])
print res



'''
除去自身外的乘积值

类似的左右遍历思路
tql这个思路
'''
class Solution(object):
    def productExceptSelf(self, nums):
        left,right = 1,1
        l = len(nums)
        res = [0]*l
        for i in range(l):
            res[i] = left   # 现在的res存放的是：每个位置上元素，其左边需要乘的值
            left *= nums[i]    # left init=1  第一个值的左边乘1啊 没问题的
        for j in range(l-1,-1,-1):
            res[j] *= right   # 这里res开始依次把每个位置上元素，其右边需要乘的值乘上
            # 所以是*=right    right init=1，最后一个值右边可不就是乘1嘛
            right *= nums[j]
        return res



s = Solution()
print s.productExceptSelf([1,2,3,4])
