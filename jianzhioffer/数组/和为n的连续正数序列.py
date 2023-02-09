# coding=utf-8
# 指定一个值s,求若干个连续数列,使得数列的和等于s
# 延用"指针"的思想. 设定small and big  若small到big大于s,则增大small,即去掉序列中较小的值,起微调效果.
# 若small到big小于s,则增大big.  每次均仅在一个方向上做调整


class Solution:
    def FindContinuousSequence(self, tsum):
        if tsum < 3:
            return []    # 因为要求至少打印2个数  那么1+2=3 sum不能小于3
        small = 1
        big = 2
        mid = (1 + tsum) >> 1  # 求这个mid值,可以使得求出多个small的可能值,即求出多组可能的序列
        thesum = 3
        res = []
        while big > small and big < tsum:
            while thesum > tsum and small < mid:  # 和大了的情况
                thesum -= small
                small += 1   # 减去这个small值 相当于small的index后移了一位 故要+1
            if thesum == tsum:
                res.append([i for i in range(small, big + 1)])

            # 这是当前和小了的情况，故把big后移
            big += 1
            thesum += big
        return res

s = Solution()
print s.FindContinuousSequence(15)