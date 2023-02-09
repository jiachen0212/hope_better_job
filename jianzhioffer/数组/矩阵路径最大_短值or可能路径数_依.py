# coding=utf-8
# 矩阵路径最大值  左上角到右下角

def fun(M):
    m = len(M)
    n = len(M[0])
    dp = [[0]*n for i in range(m)]
    dp[0] = [sum(M[0][:i+1]) for i in range(n)]
    for i in range(1, m):
        dp[i][0]  = dp[i-1][0] + M[i][0]

    for i in range(1, m):
        for j in range(1, n):  # 求最短路径的话，只需要修改成min就ok
            dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + M[i][j]
    return dp[-1][-1]

M = [[3,2,1],[5,2,1],[4,2,2],[10,9,3]]
print fun(M)
# 升级版  能走过的路径最大值有limit限制，求此限制下能走的最大值
# 从最末往前倒回去走
class Solution:
    def findlimit(self, M, i, j, limit, Sum):
        # Sum是当前的总的和
        if Sum < 0 or i < 0 or j < 0:
            return -1
        if i == 0 and j == 0:
            if Sum >= M[i][j]:  # 这两行注意理解
                return M[i][j]  # Sum >= M[i][j]是必须的，不然之前的和不就是-1
                # 已经走到[0][0]的位置了，-1就直接可以输出了
            else:
                return -1
        sum1 = sum2 = -1
        sum1 = self.findlimit(M, i-1, j, limit, Sum - M[i][j])
        sum2 = self.findlimit(M, i, j-1, limit, Sum - M[i][j])
        cur_sum = sum1 if sum1 > sum2 else sum2
        if cur_sum != -1:
            cur_sum += M[i][j]
        return cur_sum

    def main(self, M, limit):
        count = self.findlimit(M, len(M)-1, len(M[0])-1, limit, limit)
        return count

M = [[3,2,1],[5,2,1],[4,2,2],[10,9,3]]
s = Solution()
count = s.main(M, 14)
print count






####################################################
# 类似题目
# 左上角走到右下角，有多少种可能的路径   dp 就对了
class Solution(object):
    def uniquePaths(self, m, n):
        M = [[0]*n for i in range(m)]
        M[0] = [1 for i in range(n)]
        for i in range(1, m):
            M[i][0] = 1
        for i in range(1, m):
            for j in range(1, n):
                M[i][j] = M[i-1][j] + M[i][j-1]
        return M[-1][-1]
# 升级版本
# 也是左上到右下，但是中间会有1表示障碍物
# 且没有使用额外的数组开销 dp就直接原地计数
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:  # 障碍物
                    obstacleGrid[i][j] = 0
                    continue

                if i == 0 and j == 0:
                    obstacleGrid[i][j] = 1
                    continue

                if i == 0 or j == 0:
                    obstacleGrid[i][j] = obstacleGrid[i][j-1] if i == 0 else obstacleGrid[i-1][j]
                    continue

                obstacleGrid[i][j] = obstacleGrid[i][j-1] + obstacleGrid[i-1][j]
        return obstacleGrid[-1][-1]

s = Solution()
res = s.uniquePathsWithObstacles([[1]])
print res

