# coding=utf-8
# 朋友圈个数
'''
可以使用一个hasVisited数组, 依次判断每个节点, 如果其未访问,
朋友圈数加1并对该节点进行dfs搜索标记所有访问到的节点
'''

# dfs
class Solution(object):
    def dfs(self, M, i, visit):
        visit[i] = 1
        for k in range(len(M)):
            if M[i][k] == 1 and not visit[k]:
                self.dfs(M, k, visit)  # k和i有关系 把他们这一圈都visited掉

    def findCircleNum(self, M):
        if not M:
            return -1
        m = len(M)
        visit = [0]*m
        res = 0
        for i in range(m):
            if not visit[i]:
                self.dfs(M, i, visit)
                res += 1
        return res




