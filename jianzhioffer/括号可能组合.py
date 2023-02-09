# coding=utf-8
# 递归实现
# 新加入的括号'()' 可以在任意位置都ok，然后用set去重


class Solution(object):
    def generateParenthesis(self, n):
        res = set(['()'])
        for i in range(n-1):
            tmp = set()
            for r in res:
                tmp.update(set([r[:j] + '()' + r[j:] for j in range(len(r))]))
            res = tmp
        return list(res)


s = Solution()
print(s.generateParenthesis(4))