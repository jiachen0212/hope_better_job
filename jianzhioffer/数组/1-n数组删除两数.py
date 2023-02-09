# coding=utf-8
# 1-n的数组删除2个元素，O(n)时复求出这两个数

import numpy as np

nums = [4,1,5,3,7]
n = max(nums)
twoSum = (1+n)* n /2 - sum(nums)
x = sum([i**2 for i in range(1, n+1)])
y = sum([i**2 for i in nums])
twoMulSum = x - y

two_ab = twoSum**2 - twoMulSum
amb = np.sqrt(2*twoMulSum - twoSum**2)  # a-b
a = (twoSum + amb) / 2
b = twoSum - a
print a, b