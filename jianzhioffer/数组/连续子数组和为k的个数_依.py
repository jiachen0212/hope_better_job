# coding=utf-8
# 连续子数组和为k
# 要求时间复杂度O(n)


# 比较直接的 O(n^2)
nums = [1,3,2,5,-5,4]
k = 2
l = len(nums)
count = 0
for i in range(l):
    ssum = 0
    for j in range(i, l):  # 注意这里是i
        ssum += nums[j]
        if ssum == k:
            count += 1
print count


# 时间复杂度O(n)，只需要遍历一遍
# 空间复杂度也是O(n)吧  leetcode上说是O(1)
class Solution(object):
    def subarraySum(self, nums, k):
        count = 0
        if not nums:
            return count
        l = len(nums)
        Sum= 0
        dic = {}
        dic[0]=1  # 这句也不能漏且不能随便写成别的 会bug...
        for i in range(l):
            Sum += nums[i]
            if (Sum-k) in dic:
                count += dic[Sum-k]   # 这里注意是直接dic[Sum-k]
                # 重复出现了几次就加几
            if Sum not in dic:
                dic[Sum] = 0
            dic[Sum] += 1  # 这里容易出错
            # 不管s-target值如何，sum都要在dic中+=1的，所以要先判断是不是存在，然后再自加1
        return count

