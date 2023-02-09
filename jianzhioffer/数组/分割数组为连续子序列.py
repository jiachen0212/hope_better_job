# coding=utf-8
# 贪心法
'''
扫描到’1’的时候，向后寻找两个元素，凑成子序列[1,2,3]（这时1，2，3各被消耗了一个）。
接着我们就应该访问到’3’，我们又去查询以’2’结尾的（当前扫描的是3，寻找以num-1为结尾的长度>=3的序列）
有没有合法的连续序列，如果没有，那它只能向后寻找两个元素，构成至少3长度。凑出连续子序列[3,4,5]
（3，4，5个被消耗了一次），结束访问。

如果输入[1,2,3,3,4,4,5,5]，刚开始访问'1'，建立[1,2,3]，
接着访问'3'，建立[3,4,5]
接着访问'4'，由于第一步建立了[1,2,3]以4 - 1结尾的连续子序列，所以它放入，得到[1,2,3,4]
接着访问'5'，由于第一步建立了[1,2,3,4]以5 - 1结尾的连续子序列，所以它放入，得到[1,2,3,4,5]
'''
class Solution(object):
    def isPossible(self, nums):
        if not nums or len(nums)<3:
            return False

        l = len(nums)
        dic = {}
        bins = {}   # 存放以nums[i]结尾的，长度>=3的子序列
        # init
        for i in range(l):
            bins[nums[i]] = 0

        # 对数组进行遍历计数
        for i in range(l):
            dic[nums[i]] =dic.get(nums[i], 0)+1

        for i in range(l):
            if dic[nums[i]] == 0:
                continue
            dic[nums[i]] -= 1  # 个数自减1
            # 存在以nums[i-1]为末尾元素的子序列
            if nums[i]-1 in bins and bins[nums[i]-1] > 0:
                bins[nums[i]-1] -= 1
                bins[nums[i]] += 1
            # 不存在, 那就从i后面继续找2个数凑下先
            elif nums[i]+1 in dic and nums[i]+2 in dic and dic[nums[i]+1] > 0 and dic[nums[i]+2] > 0:
                dic[nums[i]+1] -= 1
                dic[nums[i]+2] -= 1
                bins[nums[i]+2] += 1   # 以i+2结尾的bins可以+=1
            else:
                return False
        return True

s = Solution()
res = s.isPossible([1,2,3,3,4,4,5,5])
print res