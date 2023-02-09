# coding=utf-8
# https://www.cnblogs.com/grandyang/p/8519566.html
'''
题目：
Input: [[1,3], [0,2], [1,3], [0,2]]
Output: true
index0处是[1,3],代表0与1，与3有连接； index1处是[0,2]，代表1与0 1与2有连接

0----1
|    |
|    |
3----2

'''



# 是否二分图
# 也是染色问题，BFS做，一直找邻找邻

'''
还是遍历整个顶点，如果未被染色，则先染色为1，然后使用 BFS 进行遍历，
将当前顶点放入队列 queue 中，然后 while 循环 queue 不为空，取出队首元素，
遍历其所有相邻的顶点，如果相邻顶点未被染色，则染成和当前顶点相反的颜色，
然后把相邻顶点加入 queue 中，否则如果当前顶点和相邻顶点颜色相同，直接返回 false，
循环退出后返回 true，
'''

class Solution():
    def isbinarygrap(self, g):
        ll = len(g)
        colors = [0]*ll
        for i in range(ll):
            if colors[i] != 0:
                continue
            colors[i] = 1
            queue = []
            queue.append(i)
            while queue:
                t = queue.pop(0)
                for j in g[t]:
                    if colors[j] == colors[t]:
                        return False
                    if colors[j] == 0:
                        colors[j] = -colors[t]
                        queue.append(j)
        return True

s = Solution()
res = s.isbinarygrap([[1,2,3], [0,2], [0,1,3], [0,2]])
print(res)

