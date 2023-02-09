# coding=utf-8
# 俄罗斯套娃信封问题

'''
输入: envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出: 3
解释: 最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
'''

'''
        O(NlogN)的做法, 按照长度升序, 同长则宽度降序排列, 然后使用O(logN)
        的最长递增子序列解法来做即可. 排序后等于把在二维(长、宽)
        上的最长递增子序列问题转换成一维(宽)上的最长递增子序列的查找, 因为对于
        长度来说已经满足递增, 只需要在宽度上也递增即为递增序列, 同长时按宽度降
        序排列的原因是避免同长时宽度小的也被列入递增序列中, 例如[3,3], [3,4]
        如果宽度也按升序来排列, [3,3]和[3,4]会形成递增序列, 而实际上不行.
'''

# nlogn复杂度
# 利用二分查找+dp思想！，时复O(nlogn)
'''
   dp[i]: 所有长度为i+1的递增子序列中, 最小的那个序列尾数.
        由定义知dp数组必然是一个递增数组, 可以用 maxL 来表示最长递增子序列的长度.
        对数组进行遍历, 依次判断每个数num将其插入dp数组相应的位置:
        1. num > dp[maxL], 表示num比所有已知递增序列的尾数都大, 将num添加入dp
           数组尾部, 并将最长递增序列长度maxL加1
        2. dp[i-1] < num <= dp[i], 只更新相应的dp[i] dp的最末元素可以变小成num
        但此时的maxlen是不变的
'''
class Solution(object):
    def maxEnvelopes(self, envelopes):
        if not envelopes:
            return 0
        l = len(envelopes)
        mmax = 0
        dp = [0] * l
        # 把原二维数组，第一维升序，第二维降序排列
        nums = sorted(envelopes, key=lambda x:(x[0], -x[1]))
        for num in nums:
            lo, hi = 0, mmax
            while lo < hi:
                mid = (lo+hi)/2
                if dp[mid] < num[1]:  # 只需要比较第二维的值就够了
                    lo = mid + 1
                else:
                    hi = mid
            dp[lo] = num[1]
            if lo == mmax:
                mmax += 1
        return mmax



