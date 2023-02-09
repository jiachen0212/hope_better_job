# coding=utf-8
# 0 1 等长的最长连续子序列
# 类似连续子序列和为k的倍数 那道题
# dict存储当前01串中01相差的个数 i j相等 则i~j之间的01个数即是先等的

'''
另一种思路：将0当做-1，将数组的各项累加起来，i:j为连续数组的充要条件是arrSum[i]=arrSum[j]
'''


class Solution(object):
    def findMaxLength(self, nums):
        N = len(nums)
        hashmap = {}
        hashmap[0] = -1
        ans = 0
        ssum = 0
        for i in range(N):
            ssum += 1 if nums[i] == 1 else -1  # 是1就加1 否则加-1 因为把0变-1了
            if ssum in hashmap:
                ans = max(ans, i - hashmap[ssum])
            else:
                hashmap[ssum] = i  # hashmap  key是sum值 value是index
        return ans

s = Solution()
res = s.findMaxLength([1,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1])
print(res)