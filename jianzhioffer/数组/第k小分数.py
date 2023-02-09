# coding=utf-8
# 第k小的分数
# 反正是排好序的数组，那就用二分吧
'''
输入: A = [1, 2, 3, 5], K = 3
输出: [2, 5]
解释:
已构造好的分数,排序后如下所示:
1/5, 1/3, 2/5, 1/2, 3/5, 2/3.
很明显第三个最小的分数是 2/5.

输入: A = [1, 7], K = 1
输出: [1, 7]
# diff

'''
# 二分巧妙在寻找的是一个mid  使得分数小于它的个数==k
# 和那个行列升序求第k小一样的思路
# 用大根堆也能做  但是时复更大

class Solution(object):
    def kthSmallestPrimeFraction(self, A, K):
        left, right = 0.0, 1.0   # 分数的最大最小值
        llen = len(A)
        while True:
            p,q = 0,1
            mid = float(left+right)/2   # 0.5
            j = 0
            count = 0
            for i in range(llen):
                # A[i] / A[j] > mid 因为除法的速度慢，所以在这里 除法转换为乘法操作
                while j < llen and A[i] > mid*A[j]:
                    j += 1
                count += (llen-j)   # llen-j  因为原数组是递增序列
                if j < llen and A[i]*q > p*A[j]:
                    p,q = A[i],A[j]
            if count == K:
                return p,q
            elif count > K:
                right = mid
            else:
                left = mid


s = Solution()
res = s.kthSmallestPrimeFraction([1,2,3,5],3)
print res