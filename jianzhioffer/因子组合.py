# coding=utf-8
# 因子组合  12:[[2,6],[2,2,3],[3,4]]
# https://www.cnblogs.com/grandyang/p/5332722.html

'''
这道题给了我们一个正整数n，让我们写出所有的因子相乘的形式，
而且规定了因子从小到大的顺序排列，那么对于这种需要列出所有的情况的题目，
通常都是用回溯法来求解的，由于题目中说明了1和n本身不能算其因子，
那么我们可以从2开始遍历到n，如果当前的数i可以被n整除，说明i是n的一个因子，
我们将其存入一位数组out中，然后递归调用n/i，此时不从2开始遍历，而是从i遍历到n/i，
停止的条件是当n等于1时，如果此时out中有因子，我们将这个组合存入结果res中
'''

class Solution(object):
    def getFactors(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        res = []
        for i in range(2, int(n ** 0.5)+1):
            if n % i == 0:
                comb = sorted([i, n / i])
                if comb not in res:
                    res.append(comb)

                # 递归
                for item in self.getFactors(n / i) :
                    comb = sorted([i] + item)
                    if comb not in res:
                        res.append(comb)

        return res

s = Solution()
ans = s.getFactors(32)
print(ans)
