# coding=utf-8
# a + b + c = 0
'''
给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

来源：力扣（LeetCode）

'''

class Solution(object):
    def threeSum(self, nums):
        ans = set([])   # set使得答案不会重复
        plus = sorted([n for n in nums if n > 0])
        plus_c = set(plus)  # 用上set是防止出现重复元素
        zeros = [n for n in nums if n == 0]
        minus = sorted([n for n in nums if n < 0])
        minus_c = set(minus)   # 用上set是防止出现重复元素

        if len(zeros) > 2:
            ans.add((0,0,0))    # 用到了set的添加  .add()
        if len(zeros) > 0:   # 放一个0进去
            for i in minus:
                if -i in plus_c:
                    ans.add((i, 0, -i))
        # 没有0， 那就++—
        n = len(plus)
        for i in range(n):
            for j in range(i+1,n):
                diff = -(plus[i]+plus[j])
                if diff in minus_c:
                    ans.add((diff,plus[i],plus[j]))
        # --+
        l_m = len(minus)
        for i in range(l_m):
            for j in range(i + 1, l_m):
                diff = minus[i] + minus[j]
                if -diff in plus_c:
                    ans.add((minus[i], minus[j], -diff))
        return list(ans)



