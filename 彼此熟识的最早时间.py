def earliestAcq(logs, n):
    logs.sort()
    parent=[i for i in range(n)]
    def find(x):
        if parent[x]!=x:
            parent[x]=find(parent[x])
        return parent[x]
    def union(x,y):
        parent[find(x)]=find(y)
    for i in range(len(logs)):
        time,a,b=logs[i][0],logs[i][1],logs[i][2]
        union(a,b)
        for i in range(n):
            find(i)
        print(parent)
        if len(set(parent))==1:
            return time
    return -1
logs, n = [[20190101,0,1],[20190104,3,4],[20190107,2,3],[20190211,1,5],[20190224,2,4],[20190301,0,3],[20190312,1,2],[20190322,4,5]], 6
print('彼此熟识的最早时间, ', earliestAcq(logs, n))
