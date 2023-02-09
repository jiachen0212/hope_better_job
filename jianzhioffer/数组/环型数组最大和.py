# coding=utf-8
# 环型数组最大和
# 思想好巧妙！！！

class Solution(object):
    def maxSubarraySumCircular(self, A):
            min_prev_sum = 0
            max_prev_sum = 0
            max_sum = -30000
            min_sum = 30000
            total_sum = 0
            flag = False
            for a in A:
                if a > 0: flag = True
                max_prev_sum = max(max_prev_sum, 0) + a
                min_prev_sum = min(min_prev_sum, 0) + a
                max_sum = max(max_prev_sum, max_sum)
                min_sum = min(min_prev_sum, min_sum)
                total_sum += a
            if flag:
                # print(total_sum, min_sum)
                # 数组所有元素的和，减去最小的和，剩下就是最大的和了
                return max(max_sum, total_sum - min_sum)
            else:
                return max_sum

s = Solution()
res = s.maxSubarraySumCircular([-1,3,-4,2])
print(res)