# coding=utf-8

class Solution(object):
    def combinationSum(self, candidates, target):
        candidates.sort()   # 升序排序先  保证每次取到的是当前的最小值
        # 递归思想！
        return self.func(candidates, target, [], min(candidates))
    def func(self, can, target, path, minV):
        res = []
        for x in range(len(can)):
            diff = target - can[x]
            if diff >= minV:
                res += self.func(can[x:], diff, path + [can[x]], can[x])
            elif diff == 0:
                res += [path + [can[x]]]

        return res

s = Solution()
res = s.combinationSum([1,4,11,15], 26)
print(res)
'''
[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 3], [1, 1, 1, 1, 1, 1, 4], [1, 1, 1, 1, 3, 3], [1, 1, 1, 1, 6], [1, 1, 1, 3, 4], [1, 1, 1, 7], [1, 1, 4, 4], [1, 3, 3, 3], [1, 3, 6], [3, 3, 4], [3, 7], [4, 6]]
'''