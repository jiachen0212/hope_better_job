# coding=utf-8
# copy的代码 总结的规律
'''
输入：10
输出：12
解释：12 = 10 * 9 / 8 + 7 - 6 * 5 / 4 + 3 - 2 * 1
'''
class Solution:
    def clumsy(self, N):
        return (N+[1,2,2,-1][N%4]) if N>4 else [7,1,2,6][N%4]