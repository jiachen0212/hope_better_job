#coding=utf-8
# 二维数组的每一行和每一列都是递增的,检测某个指定数字是否在数组中

# 逐行+行内二分
class Solution(object):
    def searchMatrix(self, matrix, target):
        if not matrix or not matrix[0]:
            return False
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            l,r = 0,n-1
            while l<=r:
                if matrix[i][r] < target:  # 这一行的最大值都比target小? 那肯定不在这行啊
                    break  # break出来，往下一行看
                else: # 这部分用二分，可以使得当矩阵维度很大的时候，时复更友好
                    mid = (l+r)/2
                    if matrix[i][mid] == target:
                        return True
                    elif matrix[i][mid] < target:
                        l = mid + 1
                    else:
                        r = mid - 1
        return False





