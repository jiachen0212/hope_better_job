# coding=utf-8
# 是否可二分
# 图问题  染色问题
# https://www.cnblogs.com/grandyang/p/10317141.html

'''
建一个大小为 (N+1) x (N+1) 的二维数组g，g[i][j] g[j][i] == 1 表示disike
开始遍历使用染色法，使用一个一维的 colors 数组，大小为 N+1，初始化是0，
由于只分两组，我们可以用1和-1来区分。
对于每个遍历到的结点，如果其还未被染色(非+-1)，我们调用递归函数对其用颜色1进行尝试染色。
在递归函数中，现将该结点染色，然后就要遍历所有跟其合不来的人，
当找到一个跟其合不来的人，首先检测其染色情况，如果此时两个人颜色相同了，说明已经在一个组里了，这就矛盾了，直接返回 false。
如果那个人还是白纸一张(0，非+-1)，我们尝试用相反的颜色去染他，如果无法成功染色，则返回 false。循环顺序退出后，返回 true
'''

#### 版本一  递归
class Solution(object):
    # cur为index，color表示这个节点被染的色
    def helper(self, g, cur, color, colors):
        colors[cur] = color
        for i in range(len(g)):
            if g[cur][i] == 1:  # g[][]==1呀，那就是你俩dislike啊
                if colors[i] == color: # 如果dislike的这个节点还染了一样的色，那就直接False额
                    return False
                # colors[i] == 0 表示这个节点还没被染色，那就尝试染成-color试试？
                if colors[i] == 0 and not self.helper(g, i, -color, colors):
                    return False
        return True

    def possibleBipartition(self, n, dislike):
        g = [[0]*(n+1) for i in range(n+1)]
        for i in range(len(dislike)):
            g[dislike[i][0]][dislike[i][1]] = 1
            g[dislike[i][1]][dislike[i][0]] = 1
        colors = [0]*(n+1)
        for i in range(n):
            # colors[i] == 0 表示这个节点还没被染色，没被分组
            # self.helper(g, i, 1, colors) 中的1表示我用1去染这个节点
            if colors[i] == 0 and not self.helper(g, i, 1, colors):
                return False
        return True




### 版本2 使用BFS
class Solution(object):
    def possibleBipartition1(self, n, dislike):
        g = [[] for i in range(n+1)]
        # 把会dislike的都强行拆开了
        for i in range(len(dislike)):
            g[dislike[i][0]].append(dislike[i][1])
            g[dislike[i][1]].append(dislike[i][0])
        colors = [0]*(n+1)
        for i in range(n):
            if colors[i] != 0:  # 染色了已经
                # 染色了的就不管了，我只关心没染色的，并对它进行BFS
                continue
            colors[i] = 1
            queue = []  # 所以这个queue也是在每一次循环内建立的，进行当前点的BFS遍历
            queue.append(i)
            while queue:
                t = queue.pop(0)
                for j in g[t]:
                    if colors[j] == colors[t]: # 啥已经是相同的颜色了，好吧那直接False
                        return False
                    if colors[j] == 0:  # 还没染色呢那快染成不一样的...
                        colors[j] = -colors[t]
                        queue.append(j) # 好了这个点可以放到队列里去，表示遍历到了已经
        return True


s = Solution()
res = s.possibleBipartition1(3, [[1,2],[1,3],[2,3]])
print(res)

