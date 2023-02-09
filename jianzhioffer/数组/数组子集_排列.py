# coding=utf-8
# 数组的子集
'''
输入: nums = [1,2,2]
输出:
[[], [1], [2], [1, 2], [2], [1, 2], [2, 2], [1, 2, 2]]
是可以有重复的  因为输入的数组中本就有两个2
'''
# 很trick
class Solution(object):
    def subsetsWithDup(self, nums):
        res = [[]]
        for i in range(len(nums)):
            for subres in res[:]:
                res.append(subres+[nums[i]])
        return res

s = Solution()
res = s.subsetsWithDup([1,2,2])
print res




# 进阶 不允许有重复
'''
输入: [1,2,2]
输出:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
先统计每个数字的频次，

之后再根据每个数字的频次来组合，如 [1，2，2，3，3，3]

得到字典｛1:1,2:2,3:3｝之后，

直接按个数组合就能得到结果也能避免重合。即0个数字的子集为1种，1个数字的子集为3种，2个数字的子集……6个数字的子集就能得到所有结果
'''
import copy
class Solution(object):
    def subsetsWithDup(self, nums):
        dic = {}
        for i in nums:
            dic[i] = dic.get(i, 0) + 1
        res = [[]]
        for i, v in dic.items():
            temp = copy.copy(res) # 浅拷贝,共享内存
            for j in res:
                temp.extend(j+[i]*(k+1) for k in range(v))
            res = temp
        return res
s = Solution()
res = s.subsetsWithDup([1,2,2])
print res