# 叶值的最小代价生成树
'''
每个节点都有 0 个或是 2 个子节点
数组arr中的值与树的中序遍历中每个叶节点的值一一对应
每个非叶节点的值等于其左子树和右子树中叶节点的最大值的乘积
求组成的树, 所有非叶节点的值之和
'''
def mctFromLeafValues(arr):
    n = len(arr)
    dp=[[0]*n for i in range(n)]
    mx=[[0]*n for i in range(n)]
    for i in range(n):
        mx[i][i]=arr[i]
    for l in range(1,n):
        for i in range(n-l):
            j = i+l
            for k in range(i,j):
                tmp = dp[i][k]+dp[k+1][j]+mx[i][k]*mx[k+1][j]
                if dp[i][j] == 0 or dp[i][j]>tmp:
                    dp[i][j]=tmp
                mx[i][j] = max(mx[i][k],mx[k+1][j],mx[i][j])       
    return dp[0][n-1]
print('mctFromLeafValues: ', mctFromLeafValues([6,2,4]))

