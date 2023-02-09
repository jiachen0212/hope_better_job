# coding=utf-8
# DFS and BFS

'''
DFS是一直找邻，然后一直遍历下去，直到把所有的节点遍历完
用在解是否存在的问题中比较多吧
'''

# 邻接矩阵 递归实现DFS
def DFS(M, i, visited):
    j = 0
    visited[i] = 1
    print(i)
    for i in range(len(M)):
        if M[i][j] and not visited[j]:
            DFS(M, j, visited)

def main(M):
    visited = [0]*len(M)
    for i in range(len(M)):
        DFS(M, i, visited)

M = [[0,1,1,1],[1,0,1,0],[1,1,0,1],[1,0,1,0]] # M是邻接矩阵
main(M)

print('=======')

# BFS主要是用来寻找最短路径
# 从某一点出发，加入队列，然后当这个点出队列，则马上把它的所有邻点加入队列，
# 重复的做这个过程，直到所有点遍历完
# 在此过程中可以有一些终止条件，可用来寻找最短距离/我们想要的目的。

def BFS(M):
    visited = [0]*len(M)
    queue = []
    for i in range(len(M)):
        if not visited[i]:
            visited[i] = 1
            queue.append(i)
            print(i)
        while queue:
            cur = queue.pop(0)
            for j in range(len(M)):
                if M[i][j] and not visited[j]:
                    print j    # 根据实际问题，这个可以做别的处理
                    queue.append(j)  # 把和节点i有关系的邻都依次的加入队列
                    visited[j] = 1   # 且做visited标记

M = [[0,1,1,1],[1,0,1,0],[1,1,0,1],[1,0,1,0]] # M是邻接矩阵
BFS(M)