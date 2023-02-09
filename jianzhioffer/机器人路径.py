#coding=utf-8

class Solution:
    def judge(self, threshold, i, j):
        # sum(map(int, str(i) + str(j)))这一句简直精髓! 直接得到坐标位置的 位和! i,j是超过1位的数也可以完美解决!
        if sum(map(int, str(i) + str(j))) <= threshold:
            return True
        else:
            return False

    def findgrid(self, threshold, rows, cols, matrix, i, j):
        count = 0
        if i<rows and j<cols and i>=0 and j>=0 and self.judge(threshold, i, j) and matrix[i][j] == 0: # matrix[i][j]==0表示没走过这一格
            matrix[i][j] = 1  # 表示已经走过了
            count = 1 + self.findgrid(threshold, rows, cols, matrix, i, j+1) \
            + self.findgrid(threshold, rows, cols, matrix, i, j-1) \
            + self.findgrid(threshold, rows, cols, matrix, i+1, j) \
            + self.findgrid(threshold, rows, cols, matrix, i-1, j)
        return count

    def movingCount(self, threshold, rows, cols):
        matrix = [[0 for i in range(cols)] for j in range(rows)]
        count = self.findgrid(threshold, rows, cols, matrix, 0, 0)
        print(matrix)
        return count

# test
s = Solution()
count = s.movingCount(4, 5, 5)
print count