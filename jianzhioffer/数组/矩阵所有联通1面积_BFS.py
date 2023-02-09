# coding=utf-8
# 所有联通的1的面积
# BFS了解一下～

ners = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
# 这个ners就是可能要search的8领域  8个方向都可能有1啊～

def fun(M, ners):
    m = len(M)
    n = len(M[0])
    mmax = 0
    visited = [[0]*n for i in range(m)]
    for i in range(m):
        for j in range(n):
            count = 0
            if M[i][j] and not visited[i][j]:
                count += 1
                # print(i,j, count, '---')
                visited[i][j] = 1
                queue = []
                queue.append([i,j])
                while queue:
                    cur = queue.pop(0)  # 好的呀，你这个节点要出队列是吧
                    for ner in ners:   # 那你的所有可能的邻都给我进队列来！
                        x = cur[0]+ner[0]
                        y = cur[1]+ner[1]
                        if (x>=0 and y>=0 and x<m and y<n \
                            and M[x][y] and not visited[x][y]):
                            count += 1
                            visited[x][y] = 1  # 遍历过了那就visited=1呗～
                            # print(x,y,'===')
                            queue.append([x,y])
            mmax = max(mmax, count)

    return mmax

# M = [[0,0,0,0],[1,1,0,1],[0,1,1,1],[0,1,0,0],[0,0,0,1]]
M = [[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]

# M = [[1,1,1,0,0],[0,1,0,1,0],[1,0,1,0,1],[0,0,0,0,1],[0,0,1,0,0]]
res = fun(M, ners)
print(res)